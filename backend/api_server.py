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

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1GB max upload size

UPLOAD_FOLDER = 'input'
OUTPUT_FOLDER = 'output'
CHUNK_SIZE = 24 * 1024 * 1024  # 24MB chunks
ALLOWED_EXTENSIONS = {
    'mp3', 'wav', 'm4a', 'flac', 'ogg', 'mp4', 'mpeg', 'mpga', 'webm',
    'm4v', 'wma', 'aac', 'm4b', 'm4p', 'm4r'
}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
transcription_progress = {}
original_files = {}  # Track original files and their chunks

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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

def split_large_file(file_path, max_size=CHUNK_SIZE):
    """Split a large file into chunks using ffmpeg"""
    try:
        file_size = os.path.getsize(file_path)
        if file_size <= max_size:
            return [file_path]  # No splitting needed
        
        print(f"Splitting file: {file_path} (size: {file_size} bytes)")
        
        # Get file extension
        file_ext = os.path.splitext(file_path)[1]
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
        chunk_duration = (max_size / file_size) * total_duration
        
        print(f"File duration: {total_duration}s, chunk duration: {chunk_duration}s")
        
        # Split the file
        chunk_files = []
        chunk_num = 0
        
        for start_time in range(0, int(total_duration), int(chunk_duration)):
            chunk_filename = f"{base_name}_chunk_{chunk_num:03d}{file_ext}"
            chunk_path = os.path.join(UPLOAD_FOLDER, os.path.basename(chunk_filename))
            
            print(f"Creating chunk {chunk_num}: {chunk_path}")
            
            # Use ffmpeg to extract chunk
            cmd = [
                'ffmpeg', '-i', file_path, '-ss', str(start_time),
                '-t', str(chunk_duration), '-c', 'copy', '-y', chunk_path
            ]
            
            result = subprocess.run(cmd, capture_output=True)
            if result.returncode == 0:
                chunk_files.append(chunk_path)
                chunk_num += 1
                print(f"Successfully created chunk {chunk_num-1}")
            else:
                error_msg = result.stderr.decode() if result.stderr else "Unknown error"
                raise Exception(f"Failed to create chunk {chunk_num}: {error_msg}")
        
        # Remove original file
        os.remove(file_path)
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
    total_chunks = len(chunks)
    
    if total_chunks > 0:
        progress_percentage = (completed_chunks / total_chunks) * 100
        
        # Update progress for all chunks of this original file
        for chunk_id in chunks:
            if chunk_id in transcription_progress:
                transcription_progress[chunk_id]['progress'] = progress_percentage
                
                # If all chunks are complete, mark as completed
                if completed_chunks == total_chunks:
                    transcription_progress[chunk_id]['status'] = 'completed'
                    # Concatenate transcripts
                    concatenate_transcripts(original_filename)

def concatenate_transcripts(original_filename):
    """Concatenate all chunk transcripts for an original file with proper SRT formatting"""
    if original_filename not in original_files:
        return
    
    chunks = original_files[original_filename]
    all_captions = []
    current_index = 1
    time_offset = 0.0
    
    # Collect all completed chunk transcripts in order
    for chunk_id in sorted(chunks, key=lambda x: transcription_progress[x].get('chunk_index', 0)):
        if (chunk_id in transcription_progress and 
            transcription_progress[chunk_id]['status'] == 'completed' and
            transcription_progress[chunk_id]['transcript']):
            
            chunk_transcript = transcription_progress[chunk_id]['transcript']
            chunk_captions = parse_srt_content(chunk_transcript)
            
            # Adjust timestamps and indices for this chunk
            for caption in chunk_captions:
                caption['start_time'] += time_offset
                caption['end_time'] += time_offset
                caption['index'] = current_index
                current_index += 1
                all_captions.append(caption)
            
            # Calculate time offset for next chunk
            if chunk_captions:
                # Get the chunk duration from ffprobe
                chunk_path = transcription_progress[chunk_id].get('chunk_path', '')
                if chunk_path and os.path.exists(chunk_path):
                    try:
                        result = subprocess.run([
                            'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
                            '-of', 'csv=p=0', chunk_path
                        ], capture_output=True, text=True)
                        
                        if result.returncode == 0:
                            chunk_duration = float(result.stdout.strip())
                            # Add the gap between chunks (if any) to the offset
                            if chunk_captions:
                                last_caption_end = chunk_captions[-1]['end_time']
                                gap_duration = chunk_duration - last_caption_end
                                if gap_duration > 0:
                                    time_offset += gap_duration
                                else:
                                    time_offset += chunk_duration
                    except:
                        # If we can't get duration, just add a small offset
                        time_offset += 1.0
    
    if all_captions:
        # Create the final SRT content
        srt_lines = []
        for caption in all_captions:
            srt_lines.append(format_srt_caption(caption))
        
        full_transcript = '\n\n'.join(srt_lines)
        
        # Save the concatenated transcript
        output_filename = f"{original_filename}.srt"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        
        with open(output_path, "w", encoding="utf-8") as output_file:
            output_file.write(full_transcript)
        
        # Update the first chunk's progress with the full transcript
        first_chunk_id = min(chunks, key=lambda x: transcription_progress[x].get('chunk_index', 0))
        if first_chunk_id in transcription_progress:
            transcription_progress[first_chunk_id]['transcript'] = full_transcript
            transcription_progress[first_chunk_id]['output_path'] = output_path

def transcribe_file(file_path, file_id):
    try:
        transcription_progress[file_id]['progress'] = 0
        transcription_progress[file_id]['status'] = 'transcribing'

        with open(file_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="srt"
            )

        cc = OpenCC('t2s')
        simplified_transcript = cc.convert(transcript)

        transcription_progress[file_id]['transcript'] = simplified_transcript
        transcription_progress[file_id]['status'] = 'completed'
        transcription_progress[file_id]['chunk_path'] = file_path

        # Update progress for original file if this is a chunk
        original_file = transcription_progress[file_id].get('original_file')
        if original_file:
            update_original_file_progress(original_file)
        else:
            # Single file (not chunked)
            filename = os.path.basename(file_path)
            output_path = os.path.join(OUTPUT_FOLDER, f"{filename}.srt")
            
            with open(output_path, "w", encoding="utf-8") as output_file:
                output_file.write(simplified_transcript)
            
            transcription_progress[file_id]['output_path'] = output_path
            transcription_progress[file_id]['progress'] = 100

        os.remove(file_path)

    except Exception as e:
        transcription_progress[file_id]['status'] = 'error'
        transcription_progress[file_id]['error'] = str(e)
        if os.path.exists(file_path):
            os.remove(file_path)

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
                filename = secure_filename(file.filename)
                file_path = os.path.join(UPLOAD_FOLDER, filename)
                file.save(file_path)

                print(f"Uploaded file: {filename}")

                # Check if file needs splitting
                file_size = os.path.getsize(file_path)
                if file_size > CHUNK_SIZE:
                    print(f"File {filename} is large ({file_size} bytes), will be split")
                    
                    # Create entry for original file first (for immediate display)
                    file_id = f"{int(time.time() * 1000)}_{filename}"
                    uploaded_files.append({
                        'id': file_id,
                        'filename': filename,
                        'path': file_path,
                        'is_original': True
                    })
                    transcription_progress[file_id] = {
                        'progress': 0,
                        'status': 'pending',
                        'error': None,
                        'transcript': None,
                        'output_path': None,
                        'is_original': True,
                        'original_filename': filename  # Store the full original filename
                    }
                    
                    # Start chunking in background
                    def chunk_file():
                        try:
                            print(f"Starting chunking process for {filename}")
                            chunk_files = split_large_file(file_path)
                            original_files[filename] = []  # Use the full filename as key
                            
                            for i, chunk_path in enumerate(chunk_files):
                                chunk_filename = os.path.basename(chunk_path)
                                chunk_id = f"{int(time.time() * 1000)}_{i}_{chunk_filename}"
                                transcription_progress[chunk_id] = {
                                    'progress': 0,
                                    'status': 'pending',
                                    'error': None,
                                    'transcript': None,
                                    'output_path': None,
                                    'original_file': filename,  # Use the full filename
                                    'chunk_index': i
                                }
                                original_files[filename].append(chunk_id)
                                print(f"Created chunk entry: {chunk_id}")
                            
                            print(f"Chunking completed for {filename}")
                        except Exception as e:
                            print(f"Chunking failed for {filename}: {str(e)}")
                            transcription_progress[file_id]['status'] = 'error'
                            transcription_progress[file_id]['error'] = f'File splitting failed: {str(e)}'
                    
                    threading.Thread(target=chunk_file).start()
                    
                else:
                    print(f"File {filename} is small ({file_size} bytes), no splitting needed")
                    # No splitting needed
                    file_id = f"{int(time.time() * 1000)}_{filename}"
                    uploaded_files.append({
                        'id': file_id,
                        'filename': filename,
                        'path': file_path
                    })
                    transcription_progress[file_id] = {
                        'progress': 0,
                        'status': 'pending',
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
                                        # Find the chunk file
                                        for filename in os.listdir(UPLOAD_FOLDER):
                                            if filename in chunk_id:
                                                file_path = os.path.join(UPLOAD_FOLDER, filename)
                                                print(f"Starting transcription for chunk {chunk_id}: {file_path}")
                                                thread = threading.Thread(
                                                    target=transcribe_file,
                                                    args=(file_path, chunk_id)
                                                )
                                                thread.start()
                                                threads.append(thread)
                                                break
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

@app.route('/download/<file_id>')
def download_transcript(file_id):
    if file_id in transcription_progress:
        file_info = transcription_progress[file_id]
        if file_info['status'] == 'completed' and file_info['output_path']:
            return send_from_directory(
                OUTPUT_FOLDER,
                os.path.basename(file_info['output_path']),
                as_attachment=True
            )

    return jsonify({'error': 'Transcript not available'}), 404

@app.route('/download-all')
def download_all_transcripts():
    try:
        import zipfile
        import tempfile

        temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')

        with zipfile.ZipFile(temp_zip.name, 'w') as zipf:
            for file_id, file_info in transcription_progress.items():
                if file_info['status'] == 'completed' and file_info['output_path']:
                    zipf.write(
                        file_info['output_path'],
                        os.path.basename(file_info['output_path'])
                    )

        return send_from_directory(
            os.path.dirname(temp_zip.name),
            os.path.basename(temp_zip.name),
            as_attachment=True,
            download_name='all_transcripts.zip'
        )

    except Exception as e:
        return jsonify({'error': f'Failed to create zip: {str(e)}'}), 500

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
