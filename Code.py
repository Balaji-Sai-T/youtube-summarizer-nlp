import re
import nltk
import torch
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import T5ForConditionalGeneration, T5Tokenizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load pre-trained T5 model and tokenizer for summarization
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

def get_youtube_transcript(video_url):
    """Extracts transcript from a YouTube video given its URL."""
    try:
        video_id = video_url.split("v=")[-1]  # Extract Video ID from URL
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = " ".join([entry["text"] for entry in transcript])
        return transcript_text
    except Exception as e:
        print(f"Error fetching transcript: {e}")
        return None

def preprocess_text(text):
    """Cleans and tokenizes the text, removing stopwords and special characters."""
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)  # Remove punctuation and special characters
    tokens = word_tokenize(text)
    filtered_words = [word for word in tokens if word not in stopwords.words('english')]
    return " ".join(filtered_words)

def generate_summary(text):
    """Summarizes the given text using the T5 model."""
    input_text = "summarize: " + text
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    
    with torch.no_grad():
        summary_ids = model.generate(inputs, max_length=100, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
    
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def main():
    """Main function to run the summarization pipeline."""
    video_url = input("Enter YouTube Video URL: ")
    transcript = get_youtube_transcript(video_url)
    
    if transcript:
        print("\nOriginal Transcript:\n", transcript[:1000], "...")  # Show only first 1000 chars
        
        processed_text = preprocess_text(transcript)
        summary = generate_summary(processed_text)
        
        print("\nGenerated Summary:\n", summary)

if __name__ == "__main__":
    main()
