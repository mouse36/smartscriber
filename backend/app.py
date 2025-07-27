
# Imports
from openai import OpenAI # Import OpenAI to interact with the OpenAI API
import dotenv # Import dotenv to load environment variables from .env file
import os # Import os to interact with the operating system
from opencc import OpenCC # Import opencc to convert text from Traditional Chinese to Simplified Chinese

# Load environment variables from .env file
dotenv.load_dotenv()

# Create a client object to interact with the OpenAI API
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Transcribe function
def transcribe(audio_file_path):
    with open(audio_file_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language="zh",
            response_format="srt")
    return transcript

#############   EXECUTION   #############

# Supported audio and video file extensions for Whisper
SUPPORTED_EXTENSIONS = (
    '.mp3', '.wav', '.m4a', '.flac', '.ogg', '.mp4', '.mpeg', '.mpga', '.webm',
    '.m4v', '.wma', '.aac', '.m4b', '.m4p', '.m4r'
)

# Get all files in input directory
input_directory = "/Users/mourecotelles/Desktop/BeeAlpha/transcription/backend/input/"
output_directory = "/Users/mourecotelles/Desktop/BeeAlpha/transcription/backend/output/"

# Get all supported files in the input directory
input_files = [f for f in os.listdir(input_directory) if f.lower().endswith(SUPPORTED_EXTENSIONS)]

for audio_file in input_files:
    audio_path = os.path.join(input_directory, audio_file)
    print(f"Starting transcription of [{audio_file}]...")
    try:
        transcript = transcribe(audio_path)
        print(f"\nTranscription done! Transcript of {audio_file}:\n\n" + transcript)

        # Convert Traditional Chinese to Simplified Chinese
        cc = OpenCC('t2s')
        simplified_transcript = cc.convert(transcript)

        # Write to file
        output_path = os.path.join(output_directory, audio_file + ".srt")
        print(f"Attempting to save this transcript file to: {output_path} ...")
        with open(output_path, "w", encoding="utf-8") as output_location:
            output_location.write(simplified_transcript)
        print(f"Yippee!!! Transcript of {audio_file} successfully saved as {audio_file}.srt at backend/output/")
    except Exception as e:
        print(f"An error occurred while processing {audio_file}: {e}")
        import traceback
        traceback.print_exc()
