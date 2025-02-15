╭──────────────────────────────────────────────────────────────────────────────╮
	                          Transcription System Starting                        
                                                                              
	 Input Directory:  C:\ivrit-ai-whisper-claude-transcriber-hakochav-v1.0.0\input   
	 Output Directory: C:\ivrit-ai-whisper-claude-transcriber-hakochav-v1.0.0\output  
	 Model:           ivrit-ai/faster-whisper-v2-d4                               
	 Device:          CUDA                                                        
╰──────────────────────────────────────────────────────────────────────────────╯
2025-02-15 17:47:26,335 - INFO - [transcribe.py:2242] - Found 1 files to process
⠹ Overall Progress ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   0% 0/1 0:00:00 -:--:--2025-02-15 17:47:26,599 - INFO - [transcribe.py:2036] - Converting C:\ivrit-ai-whisper-claude-transcriber-hakochav-v1.0.0\input\declaration_of_independence.mkv to WAV format
⠋ Overall Progress ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   0% 0/1 0:00:00 -:--:--2025-02-15 17:47:27,170 - INFO - [transcribe.py:839] - Processing audio with duration 01:54.707
⠴ Overall Progress ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   0% 0/1 0:00:01 -:--:--2025-02-15 17:47:27,656 - INFO - [transcribe.py:853] - VAD filter removed 00:36.432 of audio
⠙ Overall Progress ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   0% 0/1 0:00:04 -:--:--2025-02-15 17:47:30,469 - WARNING - [transcribe.py:569] - Gap detected between segments: 0.00 -> 10.70
⠏ Overall Progress ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   0% 0/1 0:00:07 -:--:--2025-02-15 17:47:34,352 - WARNING - [transcribe.py:569] - Gap detected between segments: 68.80 -> 74.06
⠋ Overall Progress ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   0% 0/1 0:00:08 -:--:--2025-02-15 17:47:34,423 - INFO - [transcribe.py:639] - Saved checkpoint for declaration_of_independence.mkv at position 0:01:54
2025-02-15 17:47:34,424 - INFO - [transcribe.py:872] - Successfully split text into 1 chunks
⠼ Overall Progress ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   0% 0/1 0:00:16 -:--:--2025-02-15 17:47:42,811 - INFO - [_client.py:1025] - HTTP Request: POST https://api.anthropic.com/v1/messages "HTTP/1.1 200 OK"
⠦ Overall Progress ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   0% 0/1 0:00:16 -:--:--2025-02-15 17:47:42,832 - INFO - [transcribe.py:1469] - Saved raw transcription to C:\ivrit-ai-whisper-claude-transcriber-hakochav-v1.0.0\output\declaration_of_independence.mkv_raw.txt
2025-02-15 17:47:42,833 - INFO - [transcribe.py:1484] - Saved processed output to C:\ivrit-ai-whisper-claude-transcriber-hakochav-v1.0.0\output\declaration_of_independence.mkv_processed.txt
2025-02-15 17:47:42,834 - INFO - [transcribe.py:1515] - Created full report at C:\ivrit-ai-whisper-claude-transcriber-hakochav-v1.0.0\output\declaration_of_independence.mkv_report.txt
2025-02-15 17:47:42,835 - INFO - [transcribe.py:665] - Deleted checkpoint for declaration_of_independence.mkv
  Overall Progress ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 1/1 0:00:16 0:00:00
2025-02-15 17:47:42,838 - INFO - [transcribe.py:2277] - Directory processing completed
2025-02-15 17:47:42,905 - INFO - [transcribe.py:2424] - Cleanup completed
Transcription completed successfully!