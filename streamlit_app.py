
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
        self.is_running = False


def run_async_loop(loop_instance):
    """Run the async loop in a separate thread"""
    asyncio.run(loop_instance.run())


def main():
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
    
    # st.markdown('<h1 class="icon-text"><span class="material-symbols-rounded">mic</span> Gemini Live API Controller</h1>', unsafe_allow_html=True)
    # st.markdown("Control your Gemini Live API session with audio, camera, and screen sharing")
    # st.info("✨ **Enhanced Features Enabled:** Proactive Audio & Affective Dialog for more natural conversations")

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

        st.divider()

        # Start/Stop buttons
        col1, col2 = st.columns(2)

        with col1:
            if st.button(":material/play_arrow: Start", 
                        disabled=st.session_state.session_active, 
                        use_container_width=True):
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

        with col2:
            if st.button(":material/stop_circle: Stop", 
                        disabled=not st.session_state.session_active,
                        use_container_width=True):
                if st.session_state.audio_loop:
                    st.session_state.audio_loop.stop()
                st.session_state.session_active = False
                st.session_state.audio_loop = None
                st.rerun()


        # Status indicator
        st.divider()
        if st.session_state.session_active:
            st.markdown('<div class="icon-text" style="color: #0e8945; padding: 0.5rem; background: #d4edda; border-radius: 0.5rem; margin: 0.5rem 0;"><span class="material-symbols-rounded" style="color: #0e8945;">check_circle</span> Session Active</div>', unsafe_allow_html=True)
            active_inputs = []
            if enable_audio:
                active_inputs.append('<span class="icon-text"><span class="material-symbols-rounded">mic</span> Audio</span>')
            if enable_camera:
                active_inputs.append('<span class="icon-text"><span class="material-symbols-rounded">videocam</span> Camera</span>')
            if enable_screen:
                active_inputs.append('<span class="icon-text"><span class="material-symbols-rounded">monitor</span> Screen</span>')
            if active_inputs:
                st.markdown(f'<div style="padding: 0.5rem; background: #e7f3ff; border-radius: 0.5rem; margin: 0.5rem 0;">Active: {" • ".join(active_inputs)}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="icon-text" style="color: #666; padding: 0.5rem; background: #f0f0f0; border-radius: 0.5rem; margin: 0.5rem 0;"><span class="material-symbols-rounded">radio_button_unchecked</span> Session Inactive</div>', unsafe_allow_html=True)

    # Main content area
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown('<h2 class="icon-text"><span class="material-symbols-rounded">chat</span> Send Message</h2>', unsafe_allow_html=True)
        
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
                        st.success("✅ Message sent!")
                    except Exception as e:
                        st.error(f"❌ Error sending message: {e}")
                    finally:
                        loop.close()

    with col_right:
        st.markdown('<h2 class="icon-text"><span class="material-symbols-rounded">info</span> Session Info</h2>', unsafe_allow_html=True)
        if st.session_state.audio_loop:
            st.markdown(f"""
            **Interview Type:** {st.session_state.audio_loop.interview_type}
            
            **Job Role:** {st.session_state.audio_loop.job_role}
            
            **Candidate:** {st.session_state.audio_loop.user_name}
            
            **Model:** `{MODEL.split('/')[-1]}`
            
            **Voice:** {st.session_state.audio_loop.voice_name}
            
            **Sample Rates:**
            - Send: {SEND_SAMPLE_RATE} Hz
            - Receive: {RECEIVE_SAMPLE_RATE} Hz
            """)
        else:
            st.markdown(f"""
            **Model:** `{MODEL.split('/')[-1]}`
            
            **Voice:** Zephyr (default)
            
            **Sample Rates:**
            - Send: {SEND_SAMPLE_RATE} Hz
            - Receive: {RECEIVE_SAMPLE_RATE} Hz
            """)

    # Response display area
    st.divider()
    col_responses, col_debug = st.columns([2, 1])
    
    with col_responses:
        st.markdown('<h2 class="icon-text"><span class="material-symbols-rounded">smart_toy</span> Gemini Responses</h2>', unsafe_allow_html=True)
        response_container = st.container()
        with response_container:
            # Get responses from audio loop if available
            if st.session_state.audio_loop and hasattr(st.session_state.audio_loop, 'responses'):
                for response in st.session_state.audio_loop.responses[-10:]:
                    st.text(response)
            elif not st.session_state.session_active:
                st.info("No responses yet. Start a session and send a message or speak!")
    
    with col_debug:
        st.markdown('<h2 class="icon-text"><span class="material-symbols-rounded">bug_report</span> Debug Info</h2>', unsafe_allow_html=True)
        
        if st.session_state.audio_loop:
            # Status log
            with st.expander("📊 Status Log", expanded=True):
                if st.session_state.audio_loop.status_log:
                    for status in st.session_state.audio_loop.status_log[-5:]:
                        st.caption(status)
                else:
                    st.caption("No status updates")
            
            # Error log
            with st.expander("⚠️ Error Log", expanded=False):
                if st.session_state.audio_loop.error_log:
                    for error in st.session_state.audio_loop.error_log:
                        st.error(error)
                else:
                    st.success("No errors")

    # Auto-refresh when session is active
    if st.session_state.session_active:
        import time
        time.sleep(1)
        st.rerun()


if __name__ == "__main__":
    main()
