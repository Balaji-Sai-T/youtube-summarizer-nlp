# YouTube Summarizer

## Project Overview
This project is a **YouTube video summarizer** that extracts video transcripts and generates concise summaries using **Natural Language Processing (NLP)** techniques. It utilizes **pre-trained transformer models** such as **BERT and T5** for efficient summarization, improving accessibility and user experience.

## Features
- **Automatic Transcript Extraction**: Retrieves video transcripts using the `YouTubeTranscriptApi`.
- **Text Preprocessing**: Cleans and processes raw transcript text.
- **Keyword Extraction**: Identifies important keywords to enhance summary quality.
- **Summarization using NLP Models**: Generates summaries using **T5 (Text-To-Text Transfer Transformer)**.
- **User-Friendly Execution**: Works with a simple script input for any YouTube video URL.

## Dataset and Data Processing
This project relies on video transcripts as input data. The processing pipeline includes:
1. **Transcript Retrieval**: Fetches subtitles/transcripts using `youtube-transcript-api`.
2. **Text Cleaning**: Removes unnecessary characters, timestamps, and noise.
3. **Keyword Extraction**: Identifies key phrases to retain important context.
4. **Summarization**: Uses **T5 Transformer** to generate a human-readable summary.

## Machine Learning Models Used
### **1. BERT (Bidirectional Encoder Representations from Transformers)**
- A state-of-the-art NLP model that understands text contextually.
- Used for feature extraction and pre-processing in some cases.

### **2. T5 (Text-To-Text Transfer Transformer)**
- Converts text-based tasks into a text-to-text format.
- Used to generate concise summaries from transcripts.
- Fine-tuned for summarization tasks, providing high-quality outputs.

## Installation & Setup
### **1. Install Required Dependencies**
```bash
pip install torch transformers youtube-transcript-api
```

## Run the Script
```bash
python youtube_summarizer.py --video_url <YouTube Video URL>
```

## Results & Conclusion
- **The T5 model provides accurate and meaningful summaries of YouTube transcripts.
- **The summarizer improves content consumption efficiency for long videos.
- **Future enhancements include multi-language support and interactive UI for better usability.

##Future Improvements
- **Support for multi-language summarization.
- **Integration with speech-to-text models for videos without transcripts.
- **Web or Mobile Application for easier accessibility.

This project demonstrates how NLP and Transformers can improve content accessibility and information retrieval in video-based platforms.
