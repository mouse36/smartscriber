from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import threading
import time
from werkzeug.utils import secure_filename
from openai import OpenAI
import dotenv
from opencc import OpenCC
import subprocess
import tempfile
from collections import defaultdict
import re

# Load environment variables
dotenv.load_dotenv()

# Testing mode - set to True to disable Whisper API calls for development
TESTING_MODE = False

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1GB max upload size

UPLOAD_FOLDER = 'backend/input'
OUTPUT_FOLDER = 'backend/output'
CHUNK_SIZE = 24 * 1024 * 1024  # 24MB chunks
ALLOWED_EXTENSIONS = {
    'mp3', 'wav', 'm4a', 'flac', 'ogg', 'mp4', 'mpeg', 'mpga', 'webm',
    'm4v', 'wma', 'aac', 'm4b', 'm4p', 'm4r'
}
# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
transcription_progress = {}
original_files = {}  # Track original files and their chunks

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def safe_filename(filename):
    """Create a safe filename while preserving original name for display"""
    # Keep original filename for display, but create safe version for file system
    base, ext = os.path.splitext(filename)
    # Replace problematic characters but keep Chinese characters
    safe_base = base.replace('/', '_').replace('\\', '_').replace(':', '_')
    return safe_base + ext

def parse_srt_timestamp(timestamp):
    """Parse SRT timestamp to seconds"""
    h, m, s = timestamp.split(':')
    return int(h) * 3600 + int(m) * 60 + float(s.replace(',', '.'))

def format_srt_timestamp(seconds):
    """Format seconds to SRT timestamp"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace('.', ',')

def parse_srt_content(srt_text):
    """Parse SRT content into structured format"""
    captions = []
    blocks = srt_text.strip().split('\n\n')
    
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) >= 3:
            try:
                index = int(lines[0])
                timestamp_line = lines[1]
                text = '\n'.join(lines[2:])
                
                # Parse timestamp
                start_str, end_str = timestamp_line.split(' --> ')
                start_time = parse_srt_timestamp(start_str)
                end_time = parse_srt_timestamp(end_str)
                
                captions.append({
                    'index': index,
                    'start_time': start_time,
                    'end_time': end_time,
                    'text': text
                })
            except (ValueError, IndexError):
                continue
    
    return captions

def format_srt_caption(caption):
    """Format caption back to SRT format"""
    start_str = format_srt_timestamp(caption['start_time'])
    end_str = format_srt_timestamp(caption['end_time'])
    return f"{caption['index']}\n{start_str} --> {end_str}\n{caption['text']}"

def split_large_file(file_path, max_size=CHUNK_SIZE, original_filename=None):
    """Split a large file into chunks using ffmpeg"""
    try:
        file_size = os.path.getsize(file_path)
        if file_size <= max_size:
            return [file_path]  # No splitting needed
        
        print(f"Splitting file: {file_path} (size: {file_size} bytes)")
        
        # Check if file is audio-only (much easier to handle)
        audio_check = subprocess.run([
            'ffprobe', '-v', 'quiet', '-select_streams', 'v', '-show_entries', 'stream=codec_type',
            '-of', 'csv=p=0', file_path
        ], capture_output=True, text=True)
        
        is_audio_only = audio_check.returncode != 0 or not audio_check.stdout.strip()
        print(f"File is audio-only: {is_audio_only}")
        
        # If it's a video file, extract audio first
        if not is_audio_only:
            print(f"Extracting audio from video file: {file_path}")
            audio_file_path = file_path + '.audio.m4a'
            
            # Extract audio to a temporary file
            extract_cmd = [
                'ffmpeg', '-i', file_path, '-vn', '-c:a', 'aac', '-b:a', '128k', '-y', audio_file_path
            ]
            
            extract_result = subprocess.run(extract_cmd, capture_output=True)
            if extract_result.returncode != 0:
                raise Exception(f"Failed to extract audio from video: {extract_result.stderr.decode()}")
            
            # Remove the original video file and use the audio file instead
            os.remove(file_path)
            file_path = audio_file_path
            file_size = os.path.getsize(file_path)
            print(f"Audio extracted successfully. New file size: {file_size} bytes")
            
            # Check if the audio file is now small enough
            if file_size <= max_size:
                return [file_path]  # No splitting needed
        
        # Get file extension
        file_ext = os.path.splitext(file_path)[1]
        
        # Use original filename if provided, otherwise use the file path
        if original_filename:
            # Remove extension from original filename
            original_base = os.path.splitext(original_filename)[0]
            base_name = original_base
        else:
            base_name = os.path.splitext(file_path)[0]
        
        # Calculate duration and chunk duration
        # Use ffprobe to get duration
        result = subprocess.run([
            'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
            '-of', 'csv=p=0', file_path
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            raise Exception(f"Could not determine file duration: {result.stderr}")
        
        total_duration = float(result.stdout.strip())
        
        # Calculate chunk duration based on actual duration, not file size
        # Use a conservative approach: aim for chunks around 5-10 minutes
        # This is more reliable than using file size ratios
        TARGET_CHUNK_DURATION = 600  # 10 minutes in seconds
        num_chunks = max(1, int(total_duration / TARGET_CHUNK_DURATION) + 1)
        chunk_duration = total_duration / num_chunks
        
        print(f"File duration: {total_duration}s, creating {num_chunks} chunks of ~{chunk_duration:.1f}s each")
        
        # Use a much more conservative chunk size to handle variable bitrate
        # Whisper API limit is 25MB, so we'll use 10MB to be very safe
        WHISPER_LIMIT = 10 * 1024 * 1024  # 10MB (much smaller than before)
        
        # Split the file
        chunk_files = []
        chunk_num = 0
        
        for i in range(num_chunks):
            start_time = i * chunk_duration
            end_time = min((i + 1) * chunk_duration, total_duration)
            actual_chunk_duration = end_time - start_time
            
            chunk_filename = f"{base_name}_chunk_{chunk_num:03d}{file_ext}"
            chunk_path = os.path.join(UPLOAD_FOLDER, os.path.basename(chunk_filename))
            
            print(f"Creating chunk {chunk_num}: {chunk_path} (start: {start_time:.1f}s, duration: {actual_chunk_duration:.1f}s)")
            
            # Since we're now working with audio-only files, use audio-specific settings
            cmd = [
                'ffmpeg', '-i', file_path, '-ss', str(start_time),
                '-t', str(actual_chunk_duration), '-vn', '-c:a', 'aac', '-b:a', '64k', '-y', chunk_path
            ]
            
            result = subprocess.run(cmd, capture_output=True)
            if result.returncode == 0:
                # Check if the created chunk is actually under the size limit
                chunk_size = os.path.getsize(chunk_path)
                print(f"Chunk {chunk_num} size: {chunk_size} bytes")
                
                if chunk_size > WHISPER_LIMIT:
                    print(f"WARNING: Chunk {chunk_num} is too large ({chunk_size} bytes), will be re-encoded")
                    # Re-encode with more aggressive compression to reduce size
                    temp_path = chunk_path + '.temp'
                    os.rename(chunk_path, temp_path)
                    
                    # Use more aggressive audio compression settings
                    reencode_cmd = [
                        'ffmpeg', '-i', temp_path, 
                        '-vn', '-c:a', 'aac', '-b:a', '32k',  # Lower audio bitrate
                        '-y', chunk_path
                    ]
                    
                    reencode_result = subprocess.run(reencode_cmd, capture_output=True)
                    if reencode_result.returncode == 0:
                        os.remove(temp_path)
                        new_size = os.path.getsize(chunk_path)
                        print(f"Re-encoded chunk {chunk_num} size: {new_size} bytes")
                        
                        if new_size > WHISPER_LIMIT:
                            print(f"ERROR: Chunk {chunk_num} still too large after re-encoding")
                            os.remove(chunk_path)
                            # Try even more aggressive compression as last resort
                            print(f"Attempting ultra-compression for chunk {chunk_num}")
                            ultra_compress_cmd = [
                                'ffmpeg', '-i', temp_path,
                                '-vn', '-c:a', 'aac', '-b:a', '16k',  # Very low audio bitrate
                                '-y', chunk_path
                            ]
                            
                            ultra_result = subprocess.run(ultra_compress_cmd, capture_output=True)
                            if ultra_result.returncode == 0:
                                final_size = os.path.getsize(chunk_path)
                                print(f"Ultra-compressed chunk {chunk_num} size: {final_size} bytes")
                                
                                if final_size > WHISPER_LIMIT:
                                    print(f"ERROR: Chunk {chunk_num} still too large even after ultra-compression")
                                    os.remove(chunk_path)
                                    raise Exception(f"Chunk {chunk_num} exceeds Whisper API size limit even after ultra-compression")
                            else:
                                os.remove(chunk_path)
                                raise Exception(f"Failed to ultra-compress chunk {chunk_num}: {ultra_result.stderr.decode()}")
                    else:
                        os.remove(temp_path)
                        raise Exception(f"Failed to re-encode chunk {chunk_num}: {reencode_result.stderr.decode()}")
                
                chunk_files.append(chunk_path)
                chunk_num += 1
                print(f"Successfully created chunk {chunk_num-1}")
            else:
                error_msg = result.stderr.decode() if result.stderr else "Unknown error"
                raise Exception(f"Failed to create chunk {chunk_num}: {error_msg}")
        
        print(f"Successfully split file into {len(chunk_files)} chunks")
        return chunk_files
        
    except Exception as e:
        print(f"File splitting failed: {str(e)}")
        raise Exception(f"File splitting failed: {str(e)}")

def update_original_file_progress(original_filename):
    """Update progress for the original file based on completed chunks"""
    if original_filename not in original_files:
        return
    
    chunks = original_files[original_filename]
    completed_chunks = sum(1 for chunk_id in chunks if 
                          chunk_id in transcription_progress and 
                          transcription_progress[chunk_id]['status'] == 'completed')
    error_chunks = sum(1 for chunk_id in chunks if 
                       chunk_id in transcription_progress and 
                       transcription_progress[chunk_id]['status'] == 'error')
    total_chunks = len(chunks)
    
    print(f"update_original_file_progress: {original_filename}, completed: {completed_chunks}/{total_chunks}, errors: {error_chunks}")
    
    if total_chunks > 0:
        # Find the original file ID for this original filename
        original_file_id = None
        for file_id, file_info in transcription_progress.items():
            if (file_info.get('is_original') and 
                file_info.get('original_filename') == original_filename):
                original_file_id = file_id
                break
        
        print(f"Found original file ID: {original_file_id}")
        
        # If any chunks have errors, mark the original file as error
        if error_chunks > 0:
            if original_file_id and original_file_id in transcription_progress:
                # Collect all error messages from failed chunks
                error_messages = []
                for chunk_id in chunks:
                    if (chunk_id in transcription_progress and 
                        transcription_progress[chunk_id]['status'] == 'error' and
                        transcription_progress[chunk_id].get('error')):
                        error_messages.append(transcription_progress[chunk_id]['error'])
                
                transcription_progress[original_file_id]['status'] = 'error'
                transcription_progress[original_file_id]['error'] = f'Chunk transcription failed: {"; ".join(error_messages)}'
                transcription_progress[original_file_id]['progress'] = 0
                print(f"Updated original file {original_file_id} with error: {transcription_progress[original_file_id]['error']}")
            return
        
        # Calculate progress percentage
        progress_percentage = (completed_chunks / total_chunks) * 100 if total_chunks > 0 else 0
        
        # Update status for all chunks of this original file
        for chunk_id in chunks:
            if chunk_id in transcription_progress:
                # If all chunks are complete, mark as completed
                if completed_chunks == total_chunks:
                    transcription_progress[chunk_id]['status'] = 'completed'

        # Also update the original file's status if we found it
        if original_file_id and original_file_id in transcription_progress:
            # Only mark as completed when ALL chunks are done
            transcription_progress[original_file_id]['status'] = 'transcribing' if completed_chunks < total_chunks else 'completed'
            transcription_progress[original_file_id]['progress'] = progress_percentage
            print(f"Updated original file {original_file_id}: status={transcription_progress[original_file_id]['status']}, progress={progress_percentage}%")
            
            if completed_chunks == total_chunks:
                # Set the output path for the original file
                base_name = os.path.splitext(original_filename)[0]
                output_filename = f"{base_name}.srt"
                output_path = os.path.join(OUTPUT_FOLDER, output_filename)
                
                # Check if the output file actually exists before marking as completed
                if os.path.exists(output_path):
                    transcription_progress[original_file_id]['output_path'] = output_path
                    print(f"Output file exists: {output_path}")
                else:
                    print(f"WARNING: Output file does not exist yet: {output_path}")
                    # Don't mark as completed if the file doesn't exist yet
                    transcription_progress[original_file_id]['status'] = 'transcribing'
                    transcription_progress[original_file_id]['progress'] = 95  # Almost done
                    return


def transcribe_file(file_path, file_id):
    try:
        transcription_progress[file_id]['status'] = 'transcribing'
        # Remove individual chunk progress - only original files need progress
        
        # Don't update original file progress here - wait until chunk actually completes
        # original_file = transcription_progress[file_id].get('original_file')
        # if original_file:
        #     update_original_file_progress(original_file)

        print(f"Starting transcription for {file_id}")

        with open(file_path, "rb") as audio_file:
            if TESTING_MODE:
                # In testing mode, simulate transcription without calling Whisper API
                print(f"TESTING MODE: Simulating transcription for {file_id}")
                import time
                time.sleep(2)  # Simulate processing time
                transcript = "1\n00:00:00,000 --> 00:00:05,000\n[TESTING MODE] This is a simulated transcript for testing purposes.\n\n2\n00:00:05,000 --> 00:00:10,000\nThe actual Whisper API is disabled in testing mode.\n\n3\n00:00:10,000 --> 00:00:15,000\nThis allows for free testing of the application functionality."
            else:
                # Normal mode - call Whisper API
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="srt"
                )

        cc = OpenCC('t2s')
        simplified_transcript = cc.convert(transcript)
        
        transcription_progress[file_id]['transcript'] = simplified_transcript
        transcription_progress[file_id]['status'] = 'completed'
        # Remove individual chunk progress - only original files need progress
        transcription_progress[file_id]['chunk_path'] = file_path

        # Update progress for original file if this is a chunk
        original_file = transcription_progress[file_id].get('original_file')
        if original_file:
            # Write this chunk's transcript to the output file immediately
            write_chunk_to_output(file_id, original_file, simplified_transcript)
            # Update original file progress after chunk completes
            update_original_file_progress(original_file)
        else:
            # Single file (not chunked)
            # Use original filename if available, otherwise use file path
            original_filename = transcription_progress[file_id].get('original_filename')
            if original_filename:
                # Remove file extension from original filename
                base_name = os.path.splitext(original_filename)[0]
                output_filename = f"{base_name}.srt"
            else:
                filename = os.path.basename(file_path)
                # Remove file extension from filename
                base_name = os.path.splitext(filename)[0]
                output_filename = f"{base_name}.srt"
            
            output_path = os.path.join(OUTPUT_FOLDER, output_filename)
            
            with open(output_path, "w", encoding="utf-8") as output_file:
                output_file.write(simplified_transcript)
            
            transcription_progress[file_id]['output_path'] = output_path

        os.remove(file_path)
        print(f"Completed transcription for {file_id}")

    except Exception as e:
        print(f"Error transcribing {file_id}: {str(e)}")
        transcription_progress[file_id]['status'] = 'error'
        # Remove individual chunk progress - only original files need progress
        transcription_progress[file_id]['error'] = str(e)
        if os.path.exists(file_path):
            os.remove(file_path)

def write_chunk_to_output(chunk_id, original_filename, transcript):
    """Writes a chunk transcript to the output file and concatenates with existing chunks."""
    # Remove file extension from original filename
    base_name = os.path.splitext(original_filename)[0]
    output_filename = f"{base_name}.srt"
    output_path = os.path.join(OUTPUT_FOLDER, output_filename)
    
    # Get chunk information for timestamp adjustment
    chunk_info = transcription_progress[chunk_id]
    chunk_index = chunk_info.get('chunk_index', 0)
    chunk_start_time = chunk_info.get('chunk_start_time', 0)
    
    # Parse the chunk transcript
    chunk_captions = parse_srt_content(transcript)
    
    # Adjust timestamps for this chunk
    for caption in chunk_captions:
        caption['start_time'] += chunk_start_time
        caption['end_time'] += chunk_start_time
    
    # Store this chunk's captions in the chunk_info for later concatenation
    chunk_info['processed_captions'] = chunk_captions
    
    # Check if all chunks for this original file are complete
    if original_filename in original_files:
        all_chunks = original_files[original_filename]
        completed_chunks = []
        
        for chunk_id in all_chunks:
            if chunk_id in transcription_progress:
                chunk_info = transcription_progress[chunk_id]
                if chunk_info['status'] == 'completed' and 'processed_captions' in chunk_info:
                    completed_chunks.append((chunk_info.get('chunk_index', 0), chunk_info['processed_captions']))
        
        # Create empty file if this is the first chunk
        if len(completed_chunks) == 1:
            print(f"Creating initial transcript file for {original_filename}")
            with open(output_path, "w", encoding="utf-8") as output_file:
                pass  # Create empty file
        
        # Write this chunk to the transcript file immediately
        if len(completed_chunks) > 0:
            # Sort chunks by their index to ensure correct order
            completed_chunks.sort(key=lambda x: x[0])
            
            # Combine all captions in order
            all_captions = []
            for _, captions in completed_chunks:
                all_captions.extend(captions)
            
            # Re-index all captions
            for i, caption in enumerate(all_captions):
                caption['index'] = i + 1
            
            # Write the current concatenated transcript
            with open(output_path, "w", encoding="utf-8") as output_file:
                for caption in all_captions:
                    output_file.write(format_srt_caption(caption) + "\n\n")
            
            print(f"Updated transcript file for {original_filename} with {len(all_captions)} captions ({len(completed_chunks)} chunks completed)")
        
        # If all chunks are complete, trigger the original file status update
        if len(completed_chunks) == len(all_chunks):
            print(f"All chunks completed for {original_filename}, finalizing transcript")
            # Now that the file is written, trigger the original file status update
            update_original_file_progress(original_filename)
        else:
            print(f"Chunk {chunk_index} completed for {original_filename}, waiting for {len(all_chunks) - len(completed_chunks)} more chunks")
    
    # Update the original file's output path (not the chunk's)
    # Find the original file ID for this original filename
    original_file_id = None
    for file_id, file_info in transcription_progress.items():
        if (file_info.get('is_original') and 
            file_info.get('original_filename') == original_filename):
            original_file_id = file_id
            break
    
    if original_file_id and original_file_id in transcription_progress:
        transcription_progress[original_file_id]['output_path'] = output_path
    
    print(f"Processed chunk {chunk_index} for {original_filename}")

@app.route('/upload', methods=['POST'])
def upload_files():
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400

        files = request.files.getlist('files')
        uploaded_files = []

        for file in files:
            if file.filename == '':
                continue

            if file and allowed_file(file.filename):
                original_filename = file.filename  # Keep original name for display
                safe_filename_for_fs = safe_filename(file.filename)  # Safe name for file system
                file_path = os.path.join(UPLOAD_FOLDER, safe_filename_for_fs)
                file.save(file_path)

                print(f"Uploaded file: {original_filename} (saved as {safe_filename_for_fs})")

                # Check if file needs splitting
                file_size = os.path.getsize(file_path)
                if file_size > CHUNK_SIZE:
                    print(f"File {original_filename} is large ({file_size} bytes), will be split")
                    
                    # Create entry for original file first (for immediate display)
                    file_id = f"{int(time.time() * 1000)}_{safe_filename_for_fs}"
                    uploaded_files.append({
                        'id': file_id,
                        'filename': original_filename,  # Use original name for display
                        'path': file_path,
                        'size': file_size,  # Add file size
                        'is_original': True
                    })
                    transcription_progress[file_id] = {
                        'status': 'pending',
                        'progress': 0,  # Start at 0% for original files
                        'error': None,
                        'transcript': None,
                        'output_path': None,
                        'is_original': True,
                        'original_filename': original_filename  # Store the full original filename
                    }
                    
                    # Start chunking in background
                    def chunk_file(file_path, original_filename, file_id):
                        try:
                            print(f"Starting chunking process for {original_filename}")
                            chunk_files = split_large_file(file_path, original_filename=original_filename)
                            original_files[original_filename] = []  # Use the full filename as key
                            
                            # Get the total duration of the original file for accurate timing
                            result = subprocess.run([
                                'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
                                '-of', 'csv=p=0', chunk_files[0] if chunk_files else file_path
                            ], capture_output=True, text=True)
                            
                            if result.returncode != 0:
                                raise Exception(f"Could not determine file duration: {result.stderr}")
                            
                            total_duration = float(result.stdout.strip())
                            print(f"Total file duration: {total_duration}s")
                            
                            # Calculate chunk duration based on number of chunks
                            if len(chunk_files) > 1:
                                chunk_duration = total_duration / len(chunk_files)
                                print(f"Chunk duration: {chunk_duration}s")
                            else:
                                chunk_duration = total_duration
                                print(f"Single file, no chunking needed")
                            
                            for i, chunk_path in enumerate(chunk_files):
                                chunk_filename = os.path.basename(chunk_path)
                                chunk_id = f"{int(time.time() * 1000)}_{i}_{chunk_filename}"
                                chunk_start_time = i * chunk_duration
                                
                                transcription_progress[chunk_id] = {
                                    'status': 'pending',
                                    # Remove individual chunk progress - only original files need progress
                                    'error': None,
                                    'transcript': None,
                                    'output_path': None,
                                    'original_file': original_filename,  # Use the full filename
                                    'chunk_index': i,
                                    'original_filename': original_filename,  # Store original name for display
                                    'chunk_start_time': chunk_start_time  # Store the start time of this chunk in the original file
                                }
                                original_files[original_filename].append(chunk_id)
                                print(f"Created chunk entry: {chunk_id} with start time: {chunk_start_time:.1f}s")
                            
                            # Create empty transcript file immediately when chunking starts
                            base_name = os.path.splitext(original_filename)[0]
                            output_filename = f"{base_name}.srt"
                            output_path = os.path.join(OUTPUT_FOLDER, output_filename)
                            
                            # Create empty file if it doesn't exist
                            if not os.path.exists(output_path):
                                with open(output_path, "w", encoding="utf-8") as output_file:
                                    pass  # Create empty file
                                print(f"Created empty transcript file: {output_path}")
                            
                            # Update the original file status to indicate chunking is complete
                            transcription_progress[file_id]['status'] = 'pending'
                            print(f"Chunking completed for {original_filename}")
                            
                            # Remove the original file after chunks are successfully created
                            if os.path.exists(file_path):
                                os.remove(file_path)
                                print(f"Removed original file: {file_path}")
                        except Exception as e:
                            print(f"Chunking failed for {original_filename}: {str(e)}")
                            transcription_progress[file_id]['status'] = 'error'
                            transcription_progress[file_id]['error'] = f'File splitting failed: {str(e)}'
                    
                    threading.Thread(target=chunk_file, args=(file_path, original_filename, file_id)).start()
                    
                else:
                    print(f"File {original_filename} is small ({file_size} bytes), no splitting needed")
                    # No splitting needed
                    file_id = f"{int(time.time() * 1000)}_{safe_filename_for_fs}"
                    uploaded_files.append({
                        'id': file_id,
                        'filename': original_filename,  # Use original name for display
                        'path': file_path,
                        'size': file_size  # Add file size
                    })
                    transcription_progress[file_id] = {
                        'status': 'pending',
                        'progress': 0,  # Start at 0% for single files
                        'error': None,
                        'transcript': None,
                        'output_path': None
                    }
            else:
                return jsonify({'error': f'File type not allowed: {file.filename}'}), 400

        if not uploaded_files:
            return jsonify({'error': 'No valid files uploaded'}), 400

        print(f"Upload completed: {len(uploaded_files)} files")
        return jsonify({
            'message': f'Successfully uploaded {len(uploaded_files)} files',
            'files': uploaded_files
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/transcribe', methods=['POST'])
def start_transcription():
    try:
        data = request.get_json()
        file_ids = data.get('file_ids', [])

        print(f"Starting transcription for file IDs: {file_ids}")

        if not file_ids:
            return jsonify({'error': 'No file IDs provided'}), 400

        threads = []
        for file_id in file_ids:
            if file_id in transcription_progress:
                file_info = transcription_progress[file_id]
                print(f"Processing file {file_id}: {file_info}")
                
                # Check if file has an error (e.g., chunking failed)
                if file_info.get('error'):
                    print(f"File {file_id} has error: {file_info['error']}")
                    continue
                
                if file_info['status'] == 'pending':
                    # Check if this is an original file that needs chunking
                    if file_info.get('is_original'):
                        original_filename = file_info.get('original_filename', file_info.get('filename', ''))
                        print(f"File {file_id} is original file, processing chunks for: {original_filename}")
                        
                        # Wait a bit for chunking to complete if it's still in progress
                        max_wait = 30  # Wait up to 30 seconds for chunking
                        wait_count = 0
                        while original_filename not in original_files and wait_count < max_wait:
                            print(f"Waiting for chunking to complete for {original_filename}...")
                            time.sleep(1)
                            wait_count += 1
                        
                        if original_filename in original_files:
                            print(f"Found chunks for {original_filename}: {original_files[original_filename]}")
                            # Transcribe all chunks for this original file
                            for chunk_id in original_files[original_filename]:
                                if chunk_id in transcription_progress:
                                    chunk_info = transcription_progress[chunk_id]
                                    if chunk_info['status'] == 'pending':
                                        # Find the chunk file by matching the chunk_id pattern
                                        # Look for files that contain the chunk_id in their name
                                        chunk_file_found = False
                                        for filename in os.listdir(UPLOAD_FOLDER):
                                            if chunk_id in filename or filename in chunk_id:
                                                file_path = os.path.join(UPLOAD_FOLDER, filename)
                                                print(f"Starting transcription for chunk {chunk_id}: {file_path}")
                                                thread = threading.Thread(
                                                    target=transcribe_file,
                                                    args=(file_path, chunk_id)
                                                )
                                                thread.start()
                                                threads.append(thread)
                                                chunk_file_found = True
                                                break
                                        
                                        if not chunk_file_found:
                                            print(f"Chunk file not found for {chunk_id}")
                                            # Mark chunk as error if file doesn't exist
                                            transcription_progress[chunk_id]['status'] = 'error'
                                            transcription_progress[chunk_id]['error'] = f'Chunk file not found for {chunk_id}'
                        else:
                            print(f"No chunks found for {original_filename} after waiting")
                            # Mark the original file as error if chunking failed
                            transcription_progress[file_id]['status'] = 'error'
                            transcription_progress[file_id]['error'] = 'Chunking failed or timed out'
                    else:
                        print(f"File {file_id} is single file")
                        # Single file (not chunked)
                        for filename in os.listdir(UPLOAD_FOLDER):
                            if filename in file_id:
                                file_path = os.path.join(UPLOAD_FOLDER, filename)
                                print(f"Starting transcription for single file {file_id}: {file_path}")
                                thread = threading.Thread(
                                    target=transcribe_file,
                                    args=(file_path, file_id)
                                )
                                thread.start()
                                threads.append(thread)
                                break

        print(f"Started {len(threads)} transcription threads")
        return jsonify({
            'message': f'Started transcription for {len(threads)} files',
            'file_ids': file_ids
        })

    except Exception as e:
        print(f"Transcription failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Transcription failed: {str(e)}'}), 500

@app.route('/progress/<file_id>')
def get_progress(file_id):
    if file_id in transcription_progress:
        return jsonify(transcription_progress[file_id])
    else:
        return jsonify({'error': 'File not found'}), 404

@app.route('/progress')
def get_all_progress():
    return jsonify(transcription_progress)

@app.route('/testing-mode')
def get_testing_mode():
    return jsonify({'testing_mode': TESTING_MODE})

@app.route('/download/<file_id>')
def download_transcript(file_id):
    print(f"Download request for file_id: {file_id}")
    if file_id in transcription_progress:
        file_info = transcription_progress[file_id]
        print(f"File info: {file_info}")
        
        # Check if this is an original file with chunks
        if file_info.get('is_original'):
            print(f"Original file detected: {file_info.get('original_filename')}")
            # For original files, check if we have an output file (partial or complete)
            if file_info.get('output_path') and os.path.exists(file_info['output_path']):
                print(f"Found output_path: {file_info['output_path']}")
                return send_from_directory(
                    OUTPUT_FOLDER,
                    os.path.basename(file_info['output_path']),
                    as_attachment=True
                )
            else:
                print(f"No output_path or file doesn't exist. output_path: {file_info.get('output_path')}")
                # Check if any chunks have output files
                original_filename = file_info.get('original_filename')
                if original_filename:
                    base_name = os.path.splitext(original_filename)[0]
                    output_filename = f"{base_name}.srt"
                    output_path = os.path.join(OUTPUT_FOLDER, output_filename)
                    print(f"Checking for file: {output_path}")
                    
                    if os.path.exists(output_path):
                        print(f"Found file at: {output_path}")
                        return send_from_directory(
                            OUTPUT_FOLDER,
                            output_filename,
                            as_attachment=True
                        )
                    else:
                        print(f"File does not exist: {output_path}")
        
        # For single files or chunks with direct output paths
        if file_info['status'] == 'completed' and file_info.get('output_path'):
            print(f"Single file with output_path: {file_info['output_path']}")
            if os.path.exists(file_info['output_path']):
                print(f"File exists, sending: {file_info['output_path']}")
                return send_from_directory(
                    OUTPUT_FOLDER,
                    os.path.basename(file_info['output_path']),
                    as_attachment=True
                )
            else:
                print(f"File does not exist: {file_info['output_path']}")

    print(f"File not found or no transcript available for: {file_id}")
    return jsonify({'error': 'Transcript not available'}), 404

@app.route('/download-all')
def download_all_transcripts():
    try:
        import zipfile
        import tempfile

        temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')

        with zipfile.ZipFile(temp_zip.name, 'w') as zipf:
            # Add completed single files
            for file_id, file_info in transcription_progress.items():
                if (file_info['status'] == 'completed' and 
                    file_info.get('output_path') and 
                    os.path.exists(file_info['output_path'])):
                    zipf.write(
                        file_info['output_path'],
                        os.path.basename(file_info['output_path'])
                    )
            
            # Add partial/complete transcripts from chunked files
            processed_files = set()  # Track files we've already added
            for file_id, file_info in transcription_progress.items():
                if (file_info.get('is_original') and 
                    file_info.get('original_filename') and
                    file_info['original_filename'] not in processed_files):
                    
                    original_filename = file_info['original_filename']
                    base_name = os.path.splitext(original_filename)[0]
                    output_filename = f"{base_name}.srt"
                    output_path = os.path.join(OUTPUT_FOLDER, output_filename)
                    
                    if os.path.exists(output_path):
                        zipf.write(output_path, output_filename)
                        processed_files.add(original_filename)

        return send_from_directory(
            os.path.dirname(temp_zip.name),
            os.path.basename(temp_zip.name),
            as_attachment=True,
            download_name='all_transcripts.zip'
        )

    except Exception as e:
        return jsonify({'error': f'Failed to create zip: {str(e)}'}), 500

@app.route('/remove-file/<file_id>', methods=['DELETE'])
def remove_file(file_id):
    try:
        if file_id not in transcription_progress:
            return jsonify({'error': 'File not found'}), 404
        
        file_info = transcription_progress[file_id]
        
        # If this is an original file with chunks, remove all chunks
        if file_info.get('is_original'):
            original_filename = file_info.get('original_filename', file_info.get('filename', ''))
            if original_filename in original_files:
                # Remove all chunk files
                for chunk_id in original_files[original_filename]:
                    if chunk_id in transcription_progress:
                        chunk_info = transcription_progress[chunk_id]
                        # Remove chunk file if it exists
                        if 'chunk_path' in chunk_info and os.path.exists(chunk_info['chunk_path']):
                            os.remove(chunk_info['chunk_path'])
                        # Remove chunk from progress tracking
                        del transcription_progress[chunk_id]
                
                # Remove from original_files tracking
                del original_files[original_filename]
        
        # Remove the file itself if it exists
        if 'path' in file_info and os.path.exists(file_info['path']):
            os.remove(file_info['path'])
        
        # Remove from progress tracking
        del transcription_progress[file_id]
        
        return jsonify({'message': 'File removed successfully'})
        
    except Exception as e:
        return jsonify({'error': f'File removal failed: {str(e)}'}), 500

@app.route('/cleanup', methods=['POST'])
def cleanup_files():
    try:
        for filename in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

        return jsonify({'message': 'Cleanup completed'})

    except Exception as e:
        return jsonify({'error': f'Cleanup failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
 