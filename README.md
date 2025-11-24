# Elpis - AI-Powered Interview Assistant

Elpis is an intelligent interview practice platform powered by Google's Gemini 2.5 Flash Live API with native audio capabilities. It provides real-time, voice-based interview simulations across multiple interview types, helping candidates prepare effectively for their career opportunities.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Function Calling & Reports](#function-calling--reports)
- [Project Structure](#project-structure)
- [API Reference](#api-reference)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

## Overview

Elpis addresses the challenge of effective interview preparation by providing an AI interviewer that conducts realistic, interactive voice conversations. The platform supports multiple interview formats and generates comprehensive performance reports using advanced AI function calling capabilities.

### Key Capabilities

- **Real-time Voice Interaction**: Natural conversation flow with immediate AI responses
- **Multiple Interview Types**: Technical, Behavioral, System Design, and HR/General interviews
- **Adaptive Questioning**: AI dynamically adjusts questions based on candidate responses
- **Performance Assessment**: Automated report generation with detailed feedback
- **Multimodal Input**: Support for audio, video, and screen sharing
- **Context Retention**: Maintains full conversation history throughout the session

## Features

### Interview Types

1. **Technical Interviews**
   - Programming concepts and algorithms
   - Data structures and problem-solving
   - Code optimization and best practices
   - Technology-specific questions

2. **Behavioral Interviews**
   - STAR method questioning
   - Past experience analysis
   - Teamwork and leadership scenarios
   - Conflict resolution assessment

3. **System Design Interviews**
   - Architecture design challenges
   - Scalability considerations
   - Trade-off analysis
   - Performance optimization

4. **HR/General Interviews**
   - Background and experience review
   - Career goals and motivation
   - Cultural fit assessment
   - Work style evaluation

### Core Features

- **Voice Customization**: Multiple AI voice options (Zephyr, Kore, Charon, etc.)
- **Dynamic System Instructions**: Interview behavior adapts to type and role
- **Text Interaction**: Supplementary text-based messaging alongside voice
- **Session Management**: Start/stop controls with real-time status indicators
- **Report Generation**: AI-generated performance reports via function calling
- **Download Capability**: Export reports in Markdown format

## Architecture

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           ELPIS ARCHITECTURE                             │
└─────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────┐
│                          PRESENTATION LAYER                               │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    Streamlit Web Interface                       │    │
│  ├─────────────────────────────────────────────────────────────────┤    │
│  │  • Interview Configuration Panel                                │    │
│  │  • Real-time Status Indicators                                  │    │
│  │  • Text Interaction Box                                         │    │
│  │  • Report Preview & Download                                    │    │
│  │  • Material Design Icons                                        │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                         APPLICATION LAYER                                 │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │              StreamlitAudioLoop Controller                        │   │
│  ├──────────────────────────────────────────────────────────────────┤   │
│  │                                                                   │   │
│  │  ┌─────────────────┐  ┌──────────────────┐  ┌────────────────┐  │   │
│  │  │ Session Manager │  │ Audio Processor  │  │ Tool Handler   │  │   │
│  │  ├─────────────────┤  ├──────────────────┤  ├────────────────┤  │   │
│  │  │ • Start/Stop    │  │ • Microphone In  │  │ • Function     │  │   │
│  │  │ • Config Store  │  │ • Speaker Out    │  │   Calling      │  │   │
│  │  │ • State Mgmt    │  │ • Audio Queues   │  │ • Report Gen   │  │   │
│  │  └─────────────────┘  └──────────────────┘  └────────────────┘  │   │
│  │                                                                   │   │
│  │  ┌─────────────────┐  ┌──────────────────┐  ┌────────────────┐  │   │
│  │  │ Camera Handler  │  │ Screen Capture   │  │ Message Queue  │  │   │
│  │  ├─────────────────┤  ├──────────────────┤  ├────────────────┤  │   │
│  │  │ • Frame Capture │  │ • Screen Grab    │  │ • Async Queue  │  │   │
│  │  │ • Resize/Encode │  │ • Image Encode   │  │ • Thread Safe  │  │   │
│  │  └─────────────────┘  └──────────────────┘  └────────────────┘  │   │
│  │                                                                   │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                       INTEGRATION LAYER                                   │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                   Gemini Live API Client                          │   │
│  ├──────────────────────────────────────────────────────────────────┤   │
│  │                                                                   │   │
│  │  Bidirectional Streaming Connection                              │   │
│  │  ┌────────────────────────────────────────────────────────────┐  │   │
│  │  │                                                             │  │   │
│  │  │  Upstream (Client → API)         Downstream (API → Client) │  │   │
│  │  │  ├─ Audio Stream (16kHz)         ├─ Audio Response (24kHz)│  │   │
│  │  │  ├─ Camera Frames                ├─ Text Responses         │  │   │
│  │  │  ├─ Screen Captures               ├─ Tool Calls            │  │   │
│  │  │  ├─ Text Messages                 └─ Status Updates        │  │   │
│  │  │  └─ Tool Responses                                         │  │   │
│  │  │                                                             │  │   │
│  │  └────────────────────────────────────────────────────────────┘  │   │
│  │                                                                   │   │
│  │  Configuration:                                                  │   │
│  │  • Response Modalities: AUDIO                                   │   │
│  │  • Media Resolution: MEDIUM                                     │   │
│  │  • Voice Config: Prebuilt voices                                │   │
│  │  • Context Window: 25,600 tokens → compress to 12,800          │   │
│  │  • Tools: Function declarations for report generation           │   │
│  │                                                                   │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                          AI SERVICE LAYER                                 │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │         Google Gemini 2.5 Flash Native Audio Preview             │   │
│  ├──────────────────────────────────────────────────────────────────┤   │
│  │                                                                   │   │
│  │  Core Capabilities:                                              │   │
│  │  ├─ Multimodal Understanding (Audio + Vision)                   │   │
│  │  ├─ Real-time Response Generation                               │   │
│  │  ├─ Context Window Management                                   │   │
│  │  ├─ Dynamic System Instructions                                 │   │
│  │  └─ Function Calling Support                                    │   │
│  │                                                                   │   │
│  │  Interview Logic:                                                │   │
│  │  ├─ Question Generation based on type/role                      │   │
│  │  ├─ Response Analysis & Follow-ups                              │   │
│  │  ├─ Conversation Memory & Context                               │   │
│  │  └─ Performance Assessment                                      │   │
│  │                                                                   │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                        DATA/OUTPUT LAYER                                  │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌─────────────────────┐  ┌──────────────────────┐  ┌─────────────────┐ │
│  │  Session State      │  │  Performance Reports │  │  Audio Output   │ │
│  ├─────────────────────┤  ├──────────────────────┤  ├─────────────────┤ │
│  │ • Active Status     │  │ • Markdown Format    │  │ • 24kHz Stream  │ │
│  │ • Config Data       │  │ • Structured Data    │  │ • Natural Voice │ │
│  │ • Audio Loop Ref    │  │ • Downloadable File  │  │ • PyAudio Out   │ │
│  │ • Responses Log     │  │ • Preview Display    │  │                 │ │
│  └─────────────────────┘  └──────────────────────┘  └─────────────────┘ │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘
```

### Data Flow Diagram

```
┌─────────────┐
│    User     │
└──────┬──────┘
       │
       │ 1. Configure Interview
       │    (Type, Role, Voice)
       ▼
┌──────────────────┐
│   Streamlit UI   │
└──────┬───────────┘
       │
       │ 2. Start Session
       ▼
┌────────────────────────┐
│  StreamlitAudioLoop    │
│  ┌──────────────────┐  │
│  │ Initialize Config│  │
│  │ Setup Audio I/O  │  │
│  │ Create Session   │  │
│  └──────────────────┘  │
└──────┬─────────────────┘
       │
       │ 3. Connect to Gemini Live API
       ▼
┌─────────────────────────────┐
│   Gemini Live API Session   │
│   ┌───────────────────────┐ │
│   │ System Instructions   │ │
│   │ Function Declarations │ │
│   │ Voice Configuration   │ │
│   └───────────────────────┘ │
└──────┬──────────────────────┘
       │
       │ 4. Bidirectional Streaming Starts
       │
   ┌───┴────┐
   │        │
   ▼        ▼
┌────────┐ ┌─────────┐
│ Upload │ │Download │
│ Stream │ │ Stream  │
└────┬───┘ └───┬─────┘
     │         │
     │         │ 7. Audio Response
     │         │    Text Content
     │         │    Tool Calls
     │         │
     │ 5. Audio Input (16kHz)
     │    Video Frames
     │    Text Messages
     │
     ▼         ▼
┌─────────────────────┐
│   AI Processing     │
│  ┌───────────────┐  │
│  │ Understand    │  │
│  │ Generate      │  │
│  │ Decide Tools  │  │
│  └───────────────┘  │
└──────┬──────────────┘
       │
       │ 6. Tool Call: generate_performance_report
       ▼
┌────────────────────────┐
│  Function Handler      │
│  ┌──────────────────┐  │
│  │ Parse Parameters │  │
│  │ Create MD Report │  │
│  │ Store Report     │  │
│  │ Send Response    │  │
│  └──────────────────┘  │
└──────┬─────────────────┘
       │
       │ 8. Display Report & Download Option
       ▼
┌──────────────────┐
│   Streamlit UI   │
│  ┌────────────┐  │
│  │ Preview    │  │
│  │ Download   │  │
│  └────────────┘  │
└──────────────────┘
```

## Technology Stack

### Backend
- **Python 3.13**: Core programming language
- **Google Gemini API**: AI model for conversation and analysis
  - Model: `gemini-2.5-flash-native-audio-preview-09-2025`
  - Live API for real-time streaming
- **PyAudio**: Audio input/output handling
  - Input: 16kHz, mono channel, paInt16 format
  - Output: 24kHz, mono channel, paInt16 format
- **OpenCV (cv2)**: Camera frame capture and processing
- **MSS**: Screen capture functionality
- **Asyncio**: Asynchronous event loop for non-blocking operations
- **Threading**: Background task execution

### Frontend
- **Streamlit**: Web application framework
- **Material Symbols**: Icon library for UI elements
- **Custom CSS**: Theming and styling
- **HTML/Markdown**: Content rendering

### Infrastructure
- **Streamlit Cloud**: Deployment platform
- **Git**: Version control
- **Environment Variables**: Secure API key management

## Installation

### Prerequisites

- Python 3.11 or higher
- Google Gemini API key
- Microphone and speakers
- (Optional) Webcam for video input

### Local Setup

1. **Clone the repository**
```bash
git clone https://github.com/GOPAL-YADAV-D/Elpis.git
cd Elpis
```

2. **Create virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Install system dependencies (Linux/Mac)**
```bash
# For Ubuntu/Debian
sudo apt-get install portaudio19-dev libasound2-dev libgl1

# For macOS
brew install portaudio
```

5. **Set up environment variables**
```bash
# Create .env file
echo "GEMINI_API_KEY=your_api_key_here" > .env
```

6. **Run the application**
```bash
streamlit run streamlit_app.py
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### Streamlit Configuration

The `.streamlit/config.toml` file contains custom theming:

```toml
[theme]
primaryColor = "#bb5a38"
backgroundColor = "#0e1117"
secondaryBackgroundColor = "#262730"
textColor = "#fafafa"
font = "sans serif"
```

### Audio Configuration

Adjust audio parameters in `streamlit_app.py`:

```python
FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000    # Input sample rate
RECEIVE_SAMPLE_RATE = 24000  # Output sample rate
CHUNK_SIZE = 1024            # Audio buffer size
```

## Usage

### Starting an Interview

1. **Access the Application**: Navigate to the Streamlit URL (local or deployed)
2. **Configure Settings** (Sidebar):
   - Enter your name
   - Select interview type
   - Specify job role
   - Choose input options (Audio/Camera/Screen)
   - Select AI voice
3. **Start Session**: Click the "Start" button
4. **Conduct Interview**: Speak naturally with the AI interviewer
5. **Request Report**: Ask "Can you generate my performance report?"
6. **Download**: Preview and download your assessment

### Text Interaction

While the session is active, you can:
- Type messages in the text box
- Send clarifications or additional information
- The AI responds to both voice and text seamlessly

### Stopping the Interview

- Click the "Stop" button to end the session
- Session state is preserved until a new interview starts

## Function Calling & Reports

### How It Works

Elpis uses Gemini's function calling capability to generate structured performance reports:

1. **Function Declaration**: Defined during session initialization
```python
{
    "name": "generate_performance_report",
    "description": "Generate comprehensive performance report",
    "parameters": {
        "type": "object",
        "properties": {
            "overall_assessment": {"type": "string"},
            "key_strengths": {"type": "array"},
            "areas_for_improvement": {"type": "array"},
            "technical_skills_rating": {"type": "integer"},
            # ... additional parameters
        }
    }
}
```

2. **AI Decision**: The AI autonomously decides when to call the function
3. **Tool Response**: Application handles the function call and generates Markdown
4. **User Access**: Report displayed with preview and download options

### Report Structure

Generated reports include:
- Overall Assessment (narrative)
- Key Strengths (3-5 bullet points)
- Areas for Improvement (3-5 bullet points)
- Technical Skills Rating (1-5 scale with explanation)
- Communication Rating (1-5 scale with explanation)
- Overall Rating (1-10 scale with justification)
- Final Recommendation (Proceed/Reject/Need more evaluation)

## Project Structure

```
Elpis/
├── .streamlit/
│   └── config.toml              # Streamlit theme configuration
├── .venv/                       # Virtual environment (not in repo)
├── .env                         # Environment variables (not in repo)
├── streamlit_app.py             # Main application file
├── main.py                      # CLI version (alternative interface)
├── requirements.txt             # Python dependencies
├── packages.txt                 # System dependencies for deployment
├── README.md                    # Project documentation
└── .gitignore                   # Git ignore rules
```

### Key Files

**streamlit_app.py**
- `StreamlitAudioLoop`: Main controller class
- `get_config()`: Live API configuration generator
- `main()`: Streamlit UI and application logic

**requirements.txt**
```
google-genai
opencv-python-headless
pyaudio
pillow
mss
streamlit
```

**packages.txt** (for Streamlit Cloud)
```
portaudio19-dev
libasound2-dev
libgl1
```

## API Reference

### StreamlitAudioLoop Class

```python
class StreamlitAudioLoop:
    def __init__(
        self,
        enable_camera=False,
        enable_screen=False,
        enable_audio=True,
        voice_name="Zephyr",
        user_name="Candidate",
        interview_type="Technical",
        job_role="Software Developer"
    )
```

**Methods:**
- `async run()`: Main async loop managing the session
- `async send_realtime()`: Sends queued messages to API
- `async listen_audio()`: Captures microphone input
- `async receive_audio()`: Receives and processes API responses
- `async play_audio()`: Plays audio output through speakers
- `async send_message(message)`: Sends text message to API
- `stop()`: Terminates the session

### Configuration Function

```python
def get_config(
    voice_name="Zephyr",
    system_instruction=None,
    tools=None
) -> types.LiveConnectConfig
```

Returns a configured `LiveConnectConfig` object for the Gemini Live API.

## Deployment

### Streamlit Cloud Deployment

1. **Push to GitHub**: Ensure code is in a GitHub repository
2. **Connect Streamlit Cloud**: Link your GitHub account
3. **Select Repository**: Choose the Elpis repository
4. **Configure Secrets**: Add `GEMINI_API_KEY` in Streamlit Cloud settings
5. **Deploy**: Streamlit Cloud handles the rest automatically

### Environment Variables in Streamlit Cloud

Navigate to App Settings → Secrets and add:

```toml
GEMINI_API_KEY = "your_api_key_here"
```

### Custom Domain (Optional)

Configure custom domain in Streamlit Cloud settings under "Custom domain" section.

## Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/your-feature-name`
3. **Commit changes**: `git commit -m 'Add some feature'`
4. **Push to branch**: `git push origin feature/your-feature-name`
5. **Open Pull Request**: Describe your changes clearly

### Code Style

- Follow PEP 8 guidelines for Python code
- Use descriptive variable and function names
- Add comments for complex logic
- Update documentation for new features

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Google Gemini Team for the powerful Live API
- Streamlit for the excellent web framework
- Material Design for the icon library
- Open source community for various libraries used

## Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Contact: [Your contact information]

## Roadmap

Future enhancements planned:
- Multi-language interview support
- Video analysis for body language assessment
- Company-specific question banks
- Progress tracking dashboard
- Resume parsing integration
- Code execution for technical assessments
- Interview replay and analysis
- Team collaboration features

---

Built with passion by GOPAL-YADAV-D
