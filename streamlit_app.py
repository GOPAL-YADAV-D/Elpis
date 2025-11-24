
import os
import asyncio
import base64
import io
import threading
import queue
import traceback

import cv2
import pyaudio
import PIL.Image
import mss
import streamlit as st

from google import genai
from google.genai import types

FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

MODEL = "models/gemini-2.5-flash-native-audio-preview-09-2025"


def init_client():
    """Initialize Gemini client"""
    return genai.Client(
        http_options={"api_version": "v1beta"},
        api_key=os.environ.get("GEMINI_API_KEY"),
    )


def get_config(voice_name="Zephyr", system_instruction=None):
    """Get Live API configuration"""
    return types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        media_resolution="MEDIA_RESOLUTION_MEDIUM",
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice_name)
            )
        ),
        system_instruction=system_instruction,
        context_window_compression=types.ContextWindowCompressionConfig(
            trigger_tokens=25600,
            sliding_window=types.SlidingWindow(target_tokens=12800),
        ),
    )


class StreamlitAudioLoop:
    def __init__(self, enable_camera=False, enable_screen=False, enable_audio=True, voice_name="Zephyr", user_name="Candidate", interview_type="Technical", job_role="Software Developer"):
        self.enable_camera = enable_camera
        self.enable_screen = enable_screen
        self.enable_audio = enable_audio
        self.voice_name = voice_name
        self.user_name = user_name
        self.interview_type = interview_type
        self.job_role = job_role
        
        self.audio_in_queue = None
        self.out_queue = None
        self.session = None
        self.audio_stream = None
        self.audio_output_stream = None
        self.is_running = False
        self.pya = pyaudio.PyAudio()
        self.client = init_client()
        
        # Generate dynamic system instruction based on interview type
        system_instruction = self._generate_system_instruction()
        
        self.config = get_config(voice_name, system_instruction)
        self.error_log = []
        self.status_log = []
        self.conversation_history = []  # Track full conversation
        self.interview_start_time = None
        self.interview_end_time = None
    
    def _generate_system_instruction(self):
        """Generate system instruction based on interview type and role"""
        base_intro = f"""You are a friendly but professional {self.interview_type.lower()} interviewer.
Your goal is to interview {self.user_name} for a {self.job_role} position.
Start by greeting {self.user_name} by name and asking them to introduce themselves."""
        
        if self.interview_type == "Technical":
            specific_instructions = f"""Then, ask relevant technical questions based on the {self.job_role} role.
For software development roles, focus on programming concepts, algorithms, data structures, and system design.
For data science roles, focus on statistics, machine learning, data analysis, and tools.
Listen to their answers, provide brief constructive feedback if needed, and move to the next question."""
        elif self.interview_type == "Behavioral":
            specific_instructions = """Then, ask behavioral questions using the STAR method (Situation, Task, Action, Result).
Ask about past experiences, teamwork, conflict resolution, leadership, and problem-solving.
Listen actively and ask follow-up questions to understand their experiences better."""
        elif self.interview_type == "System Design":
            specific_instructions = """Then, present system design problems appropriate for the role.
Ask them to design scalable systems, discuss trade-offs, and explain their architectural decisions.
Probe deeper into scalability, reliability, and performance considerations."""
        elif self.interview_type == "HR/General":
            specific_instructions = """Then, ask questions about their background, career goals, motivation, and cultural fit.
Discuss their expectations, work style, and why they're interested in this role.
Keep the conversation warm and welcoming while gathering important information."""
        else:
            specific_instructions = """Then, ask relevant questions appropriate for the interview type and role.
Listen to their answers and provide appropriate feedback."""
        
        return f"""{base_intro}\n{specific_instructions}\nKeep the conversation engaging and interactive."""

    def _get_frame(self, cap):
        ret, frame = cap.read()
        if not ret:
            return None
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = PIL.Image.fromarray(frame_rgb)
        img.thumbnail([1024, 1024])

        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)

        mime_type = "image/jpeg"
        image_bytes = image_io.read()
        return {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}

    async def get_frames(self):
        cap = await asyncio.to_thread(cv2.VideoCapture, 0)
        while self.is_running:
            frame = await asyncio.to_thread(self._get_frame, cap)
            if frame is None:
                break
            await asyncio.sleep(1.0)
            await self.out_queue.put(frame)
        cap.release()

    def _get_screen(self):
        sct = mss.mss()
        monitor = sct.monitors[0]
        i = sct.grab(monitor)

        mime_type = "image/jpeg"
        image_bytes = mss.tools.to_png(i.rgb, i.size)
        img = PIL.Image.open(io.BytesIO(image_bytes))

        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)

        image_bytes = image_io.read()
        return {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}

    async def get_screen(self):
        while self.is_running:
            frame = await asyncio.to_thread(self._get_screen)
            if frame is None:
                break
            await asyncio.sleep(1.0)
            await self.out_queue.put(frame)

    async def send_realtime(self):
        while self.is_running:
            try:
                msg = await asyncio.wait_for(self.out_queue.get(), timeout=0.5)
                await self.session.send(input=msg)
            except asyncio.TimeoutError:
                continue

    async def listen_audio(self):
        try:
            self.status_log.append("🎤 Starting audio input...")
            mic_info = self.pya.get_default_input_device_info()
            self.status_log.append(f"🎤 Using microphone: {mic_info.get('name', 'Unknown')}")
            
            self.audio_stream = await asyncio.to_thread(
                self.pya.open,
                format=FORMAT,
                channels=CHANNELS,
                rate=SEND_SAMPLE_RATE,
                input=True,
                input_device_index=mic_info["index"],
                frames_per_buffer=CHUNK_SIZE,
            )
            self.status_log.append("✅ Audio input stream opened")
            
            kwargs = {"exception_on_overflow": False}
            while self.is_running:
                try:
                    data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, **kwargs)
                    await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})
                except Exception as e:
                    if self.is_running:
                        error_msg = f"Audio input error: {e}"
                        self.error_log.append(error_msg)
                    break
        except Exception as e:
            error_msg = f"Failed to initialize audio input: {e}"
            self.error_log.append(error_msg)

    async def receive_audio(self):
        self.status_log.append("📥 Starting to receive audio...")
        while self.is_running:
            try:
                turn = self.session.receive()
                async for response in turn:
                    if not self.is_running:
                        break
                    if data := response.data:
                        self.audio_in_queue.put_nowait(data)
                        self.status_log.append(f"🔊 Received audio chunk: {len(data)} bytes")
                        continue
                    if text := response.text:
                        self.status_log.append(f"💬 Received text: {text[:50]}...")
                        # Store text for display (use a thread-safe list)
                        if not hasattr(self, 'responses'):
                            self.responses = []
                        self.responses.append(text)
                        # Track in conversation history
                        self.conversation_history.append({"role": "interviewer", "text": text})

                while not self.audio_in_queue.empty():
                    self.audio_in_queue.get_nowait()
            except Exception as e:
                if self.is_running:
                    error_msg = f"Receive audio error: {e}\n{traceback.format_exc()}"
                    self.error_log.append(error_msg)
                break

    async def play_audio(self):
        try:
            self.status_log.append("🔊 Opening audio output stream...")
            self.audio_output_stream = await asyncio.to_thread(
                self.pya.open,
                format=FORMAT,
                channels=CHANNELS,
                rate=RECEIVE_SAMPLE_RATE,
                output=True,
            )
            self.status_log.append("✅ Audio output stream opened")
            
            audio_played = False
            while self.is_running:
                try:
                    bytestream = await asyncio.wait_for(self.audio_in_queue.get(), timeout=0.5)
                    await asyncio.to_thread(self.audio_output_stream.write, bytestream)
                    if not audio_played:
                        self.status_log.append("🎵 Playing audio...")
                        audio_played = True
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    if self.is_running:
                        error_msg = f"Audio playback error: {e}\n{traceback.format_exc()}"
                        self.error_log.append(error_msg)
                    break
        except Exception as e:
            error_msg = f"Failed to initialize audio output: {e}\n{traceback.format_exc()}"
            self.error_log.append(error_msg)

    async def send_message(self, message):
        """Send a text message to the model"""
        if self.session and self.is_running:
            # Track candidate message in conversation history
            self.conversation_history.append({"role": "candidate", "text": message})
            await self.session.send(input=message, end_of_turn=True)

    async def run(self):
        try:
            self.status_log.append("🔌 Connecting to Gemini Live API...")
            async with (
                self.client.aio.live.connect(model=MODEL, config=self.config) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session
                self.audio_in_queue = asyncio.Queue()
                self.out_queue = asyncio.Queue(maxsize=5)
                self.is_running = True
                self.responses = []
                self.interview_start_time = asyncio.get_event_loop().time()
                
                self.status_log.append("✅ Connected to Gemini Live API")

                tg.create_task(self.send_realtime())
                
                if self.enable_audio:
                    tg.create_task(self.listen_audio())
                
                if self.enable_camera:
                    tg.create_task(self.get_frames())
                elif self.enable_screen:
                    tg.create_task(self.get_screen())

                tg.create_task(self.receive_audio())
                tg.create_task(self.play_audio())

                # Keep running until stopped
                while self.is_running:
                    await asyncio.sleep(0.1)

                raise asyncio.CancelledError("Session stopped")

        except asyncio.CancelledError:
            self.status_log.append("⏹️ Session stopped by user")
        except Exception as e:
            error_msg = f"Error in run loop: {e}\n{traceback.format_exc()}"
            self.error_log.append(error_msg)
        finally:
            self.is_running = False
            if self.audio_stream:
                self.audio_stream.close()
            if self.audio_output_stream:
                self.audio_output_stream.close()

    def stop(self):
        """Stop the session"""
        if self.interview_start_time and not self.interview_end_time:
            import time
            self.interview_end_time = time.time()
        self.is_running = False
    
    async def generate_performance_report(self):
        """Generate a detailed performance report using Gemini"""
        
        # Calculate interview duration
        if self.interview_start_time and self.interview_end_time:
            duration_seconds = int(self.interview_end_time - self.interview_start_time)
            duration_minutes = duration_seconds // 60
            duration_display = f"{duration_minutes} minutes {duration_seconds % 60} seconds"
        else:
            duration_display = "Unknown"
        
        # Build conversation transcript from available data
        interviewer_responses = getattr(self, 'responses', [])
        text_messages = self.conversation_history if hasattr(self, 'conversation_history') and self.conversation_history else []
        
        # If we have no data at all, return error
        if not interviewer_responses and not text_messages:
            return {
                "error": "No interview data available. The interview may have been too short or no conversation was recorded.",
                "duration": duration_display,
                "interview_type": self.interview_type,
                "job_role": self.job_role,
                "candidate_name": self.user_name
            }
        
        # Build transcript from available data
        transcript_parts = []
        
        # Add interviewer responses
        if interviewer_responses:
            transcript_parts.append("=== Interviewer Questions and Comments ===")
            for i, response in enumerate(interviewer_responses, 1):
                transcript_parts.append(f"\nInterviewer [{i}]: {response}")
        
        # Add text messages if any
        if text_messages:
            transcript_parts.append("\n\n=== Text Messages Exchanged ===")
            for msg in text_messages:
                role = msg.get('role', 'unknown').title()
                text = msg.get('text', '')
                transcript_parts.append(f"\n{role}: {text}")
        
        transcript = "\n".join(transcript_parts)
        
        # Create analysis prompt
        analysis_prompt = f"""You are an expert interview analyst. Analyze the following {self.interview_type} interview for a {self.job_role} position.

Interview Duration: {duration_display}
Candidate Name: {self.user_name}

Note: This is a voice-based interview. The transcript below shows the interviewer's questions and comments. The candidate responded verbally (audio not transcribed).

Interview Content:
{transcript}

Based on the interviewer's questions, comments, and the interview duration, provide a comprehensive performance report with the following sections:

1. **Interview Summary** (1 paragraph describing what was covered based on the questions asked)

2. **Questions & Topics Covered** (List the main topics and questions discussed)

3. **Interview Quality Assessment** (Rate the interview structure and question quality 1-5)

4. **Estimated Engagement Level** (Based on interview length and question flow: High/Medium/Low with explanation)

5. **Recommendations for Next Steps**
   - If interview was comprehensive: Recommend proceeding with technical assessment or next round
   - If interview was brief: Recommend follow-up interview
   - Suggest what additional areas should be explored

6. **Overall Interview Rating** (1-10 scale based on interview completeness and question quality)

Note: Since we don't have candidate audio transcriptions, focus your analysis on the interview structure, questions asked, and overall process quality."""
        
        try:
            # Use a simple generate request for the analysis
            response = await self.client.aio.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents=analysis_prompt
            )
            return {
                "duration": duration_display,
                "interview_type": self.interview_type,
                "job_role": self.job_role,
                "candidate_name": self.user_name,
                "analysis": response.text,
                "transcript": transcript,
                "num_questions": len(interviewer_responses)
            }
        except Exception as e:
            import traceback
            return {
                "error": f"Failed to generate report: {str(e)}\n{traceback.format_exc()}",
                "transcript": transcript,
                "duration": duration_display,
                "interview_type": self.interview_type,
                "job_role": self.job_role,
                "candidate_name": self.user_name
            }


def run_async_loop(loop_instance):
    """Run the async loop in a separate thread"""
    asyncio.run(loop_instance.run())


def main():
    st.markdown("""
    <style>
    .round-btn > button {
        border-radius: 50% !important;
        height: 100px !important;
        width: 100px !important;
        padding: 0 !important;
        font-size: 28px !important;
        text-align: center !important;
    }
    </style>
    """, unsafe_allow_html=True)


    st.set_page_config(page_title="Elpis: Interview Assistant", layout="wide", page_icon="")
    
    # Add Material Symbols font and custom CSS
    st.markdown("""
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200" />
    <style>
        .material-symbols-rounded {
            font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24;
            vertical-align: middle;
            font-size: 1.2em;
        }
        .icon-text {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
        }
    </style>
    """, unsafe_allow_html=True)

    # Check for API key
    if not os.environ.get("GEMINI_API_KEY"):
        st.error("⚠️ GEMINI_API_KEY environment variable not set!")
        st.info("Please set your API key: `set GEMINI_API_KEY=your_key_here` (Windows)")
        return

    # Initialize session state
    if "session_active" not in st.session_state:
        st.session_state.session_active = False
    if "audio_loop" not in st.session_state:
        st.session_state.audio_loop = None
    if "thread" not in st.session_state:
        st.session_state.thread = None
    if "gemini_responses" not in st.session_state:
        st.session_state.gemini_responses = []
    if "show_report" not in st.session_state:
        st.session_state.show_report = False
    if "performance_report" not in st.session_state:
        st.session_state.performance_report = None

    # Sidebar controls
    with st.sidebar:
        st.markdown('<h2 class="icon-text"><span class="material-symbols-rounded">settings</span> Interview Settings</h2>', unsafe_allow_html=True)
        
        # Interview Configuration
        st.markdown('<h3 class="icon-text"><span class="material-symbols-rounded">target</span> Interview Configuration</h3>', unsafe_allow_html=True)
        user_name = st.text_input(":material/person: Your Name", value="Candidate", disabled=st.session_state.session_active, key="user_name_input")


        
        interview_types = ["Technical", "Behavioral", "System Design", "HR/General"]

        selected_interview_type = st.selectbox(":material/grading: Interview Type",interview_types,index=0,disabled=st.session_state.session_active,help="Choose the type of interview you want to practice")

        
        job_role = st.text_input(":material/work: Job Role", value="Software Developer", disabled=st.session_state.session_active, help="Enter the specific role (e.g., Python Developer, Data Scientist, Frontend Engineer)")

        
        st.divider()
        
      # Input options header
        st.markdown('<h3 class="icon-text"><span class="material-symbols-rounded">tune</span> Input Options</h3>', unsafe_allow_html=True)

        # Toggles
        enable_audio = st.toggle(":material/mic: Enable Audio Input", value=True, disabled=st.session_state.session_active)
        enable_camera = st.toggle(":material/photo_camera: Enable Camera", value=False, disabled=st.session_state.session_active)
        enable_screen = st.toggle(":material/screen_share: Enable Screen Sharing", value=False, disabled=st.session_state.session_active)

        # Voice selectbox
        voice_options = ["Zephyr", "Puck", "Charon", "Kore", "Fenrir", "Aoede"]
        selected_voice = st.selectbox(":material/record_voice_over: Voice", voice_options, index=0, disabled=st.session_state.session_active)

        if enable_camera and enable_screen:
            st.warning("⚠️ Only one video source can be active at a time. Camera will be prioritized.")
            enable_screen = False

        # Status indicator
        st.divider()
        if st.session_state.session_active:
            st.markdown('''
                <div class="icon-text" style="
                    color: #0e8945; 
                    padding: 0.75rem 1rem; 
                    background: #d4edda; 
                    border-radius: 0.5rem; 
                    margin: 0.75rem 0;
                    border-left: 4px solid #0e8945;
                ">
                    <span class="material-symbols-rounded" style="color: #0e8945;">check_circle</span> 
                    <strong>Session Active</strong>
                </div>
            ''', unsafe_allow_html=True)
            active_inputs = []
            if enable_audio:
                active_inputs.append('<span class="icon-text"><span class="material-symbols-rounded">mic</span> Audio</span>')
            if enable_camera:
                active_inputs.append('<span class="icon-text"><span class="material-symbols-rounded">videocam</span> Camera</span>')
            if enable_screen:
                active_inputs.append('<span class="icon-text"><span class="material-symbols-rounded">monitor</span> Screen</span>')
            if active_inputs:
                st.markdown(f'''
                    <div style="
                        padding: 0.75rem 1rem; 
                        background: #e7f3ff; 
                        border-radius: 0.5rem; 
                        margin: 0.5rem 0;
                        border-left: 4px solid #0068c9;
                    ">
                        <strong>Active:</strong> {" • ".join(active_inputs)}
                    </div>
                ''', unsafe_allow_html=True)
        else:
            st.markdown('''
                <div class="icon-text" style="
                    color: #666; 
                    padding: 0.75rem 1rem; 
                    background: #f0f0f0; 
                    border-radius: 0.5rem; 
                    margin: 0.75rem 0;
                    border-left: 4px solid #999;
                ">
                    <span class="material-symbols-rounded">radio_button_unchecked</span> 
                    <strong>Session Inactive</strong>
                </div>
            ''', unsafe_allow_html=True)


    left_col, _, _ = st.columns([1, 3, 3])

    with left_col:
        if not st.session_state.session_active:
            # Play Button
            if st.button(
                ":material/play_arrow: Start",
                key="start_btn",
                use_container_width=False,
                help="Start Interview",
                type="primary",
                icon=None,
                on_click=None,
                kwargs=None,
                args=None
            ):
                st.session_state.audio_loop = StreamlitAudioLoop(
                    enable_camera=enable_camera,
                    enable_screen=enable_screen,
                    enable_audio=enable_audio,
                    voice_name=selected_voice,
                    user_name=user_name,
                    interview_type=selected_interview_type,
                    job_role=job_role
                )
                st.session_state.thread = threading.Thread(
                    target=run_async_loop,
                    args=(st.session_state.audio_loop,),
                    daemon=True
                )
                st.session_state.thread.start()
                st.session_state.session_active = True
                st.session_state.gemini_responses = []
                st.rerun()

            # Apply circular style
            st.markdown('<div class="round-btn"></div>', unsafe_allow_html=True)

        else:
            # Stop Button
            if st.button(
                ":material/stop_circle: Stop",
                key="stop_btn",
                use_container_width=False,
                help="Stop Interview",
                type="secondary"
            ):
                if st.session_state.audio_loop:
                    st.session_state.audio_loop.stop()
                st.session_state.session_active = False
                # Store the audio loop instance before clearing it so we can generate report
                st.session_state.last_audio_loop = st.session_state.audio_loop
                st.session_state.audio_loop = None
                st.rerun()

            # Apply circular style
            st.markdown('<div class="round-btn"></div>', unsafe_allow_html=True)
    
    # Performance Report Button (show after interview ends)
    if not st.session_state.session_active and st.session_state.get("last_audio_loop"):
        st.markdown("---")
        col_report1, col_report2, col_report3 = st.columns([1, 2, 1])
        with col_report2:
            # Show some debug info
            last_loop = st.session_state.last_audio_loop
            num_responses = len(getattr(last_loop, 'responses', []))
            st.caption(f"📊 Interview ended. {num_responses} interviewer responses recorded.")
            
            if st.button(
                ":material/assessment: Generate Interview Report",
                key="generate_report_btn",
                use_container_width=True,
                type="primary"
            ):
                with st.spinner("Analyzing interview..."):
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        report = loop.run_until_complete(st.session_state.last_audio_loop.generate_performance_report())
                        st.session_state.performance_report = report
                        st.session_state.show_report = True
                    except Exception as e:
                        st.error(f"❌ Error generating report: {e}")
                        import traceback
                        st.code(traceback.format_exc())
                    finally:
                        loop.close()
                        st.rerun()

    # Display Performance Report
    if st.session_state.show_report and st.session_state.performance_report:
        st.markdown("---")
        st.markdown('<h2 class="icon-text" style="text-align: center;"><span class="material-symbols-rounded">assessment</span> Interview Report</h2>', unsafe_allow_html=True)
        
        report = st.session_state.performance_report
        
        if "error" in report:
            st.error("⚠️ " + report["error"])
            
            # Still show what we have
            if report.get("duration"):
                st.info(f"**Duration:** {report['duration']}")
            if report.get("transcript"):
                with st.expander("📝 View Available Data"):
                    st.text(report["transcript"])
        else:
            # Report Header
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Candidate", report["candidate_name"])
            with col2:
                st.metric("Role", report["job_role"])
            with col3:
                st.metric("Duration", report["duration"])
            with col4:
                st.metric("Questions Asked", report.get("num_questions", "N/A"))
            
            st.info(f"**Interview Type:** {report['interview_type']}")
            
            # Analysis Section
            st.markdown("### 📊 Interview Analysis")
            st.markdown(report["analysis"])
            
            # Transcript Section (Collapsible)
            with st.expander("📝 View Interview Content"):
                st.text(report["transcript"])
            
            # Action Buttons
            st.markdown("---")
            col_action1, col_action2 = st.columns(2)
            with col_action1:
                if st.button(":material/download: Download Report", use_container_width=True):
                    # Create downloadable report
                    report_text = f"""
PERFORMANCE REPORT
==================

Candidate: {report['candidate_name']}
Position: {report['job_role']}
Interview Type: {report['interview_type']}
Duration: {report['duration']}

{report['analysis']}

FULL TRANSCRIPT
===============

{report['transcript']}
"""
                    st.download_button(
                        label="Click to Download",
                        data=report_text,
                        file_name=f"interview_report_{report['candidate_name'].replace(' ', '_')}.txt",
                        mime="text/plain"
                    )
            
            with col_action2:
                if st.button(":material/refresh: Start New Interview", use_container_width=True):
                    st.session_state.show_report = False
                    st.session_state.performance_report = None
                    st.session_state.last_audio_loop = None
                    st.rerun()

    # Main content area
    # col_left, col_right = st.columns([2, 1])

    # with col_left:
    
    # Create centered container with reduced width
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<h2 class="icon-text" style="text-align: center;"><span class="material-symbols-rounded">chat</span> Text Interaction</h2>', unsafe_allow_html=True)
        
        # Text input for sending messages
        with st.form(key="message_form", clear_on_submit=True):
            user_message = st.text_area(
                "Type your message:",
                placeholder="Enter a message to send to Gemini...",
                disabled=not st.session_state.session_active,
                height=100
            )
            submit_button = st.form_submit_button(
                "Send Message",
                disabled=not st.session_state.session_active,
                use_container_width=True
            )
            
            if submit_button and user_message:
                if st.session_state.audio_loop:
                    # Create a new event loop for this operation
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        loop.run_until_complete(st.session_state.audio_loop.send_message(user_message))
                        st.success("Message sent!")
                    except Exception as e:
                        st.error(f"Error sending message: {e}")
                    finally:
                        loop.close()

    # Auto-refresh when session is active
    if st.session_state.session_active:
        import time
        time.sleep(1)
        st.rerun()


if __name__ == "__main__":
    main()
