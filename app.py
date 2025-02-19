import streamlit as st
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import Tool
import os
from dotenv import load_dotenv
import tempfile
from PyPDF2 import PdfReader

# Initialize environment
load_dotenv()

# Initialize Groq and Google Serper
groq_chat = ChatGroq(temperature=0.7, 
                    groq_api_key=st.secrets["GROQ_API_KEY"], 
                    model_name="deepseek-r1-distill-llama-70b")

serper = GoogleSerperAPIWrapper(serper_api_key=os.getenv("SERPER_API_KEY"))

# Create search tool
search_tool = Tool(
    name="ResourceSearch",
    func=serper.run,
    description="Search for up-to-date educational resources and learning materials"
)

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content="""You are an expert study planner assistant. Your tasks:
1. Engage in a structured conversation to fully understand the student's needs. IMP:Ask only one question(this is compulsory) at a time and collect:
    Don't ask questions together, one at a time question should be asked
   - Subject of study
   - Learning goals (e.g., exam preparation, skill development)
   - Daily study time available
   - Current skill level (Beginner/Intermediate/Advanced)
   - Preferred learning style (Visual, Auditory, Reading/Writing, Kinesthetic)
2. After gathering all details, generate a personalized study plan that includes:
   - A topic-wise breakdown with clear subtopics
   - A detailed daily/weekly schedule with time allocations
   - Practical study techniques and revision strategies
 Dont hallucinate
3. you have to Use the ResourceSearch tool to find books, videos, online materials.Don't hallucinate
"""),
        AIMessage(content="👋 Welcome! I'm your AI Study Assistant. Let's create your perfect learning plan! Could you first tell me what subject you want to focus on?")
    ]

if "collected_data" not in st.session_state:
    st.session_state.collected_data = {}
if "plan_generated" not in st.session_state:
    st.session_state.plan_generated = False

def generate_response(user_input):
    st.session_state.messages.append(HumanMessage(content=user_input))
    
    # Generate response with resource search capability
    response = groq_chat.invoke([
        *st.session_state.messages,
        SystemMessage(content=f"Available tool: {search_tool.description}")
    ])
    
    # Search for resources if needed
    if "[SEARCH]" in response.content:
        search_query = response.content.split("[SEARCH]")[1].strip()
        resources = search_tool.run(search_query)
        response.content = response.content.replace(f"[SEARCH]{search_query}", 
                                                   f"Found resources: {resources}")

    if "here is your study plan" in response.content.lower():
        st.session_state.plan_generated = True
    else:
        st.session_state.messages.append(response)
    
    return response.content

# Main app interface
st.title("Smart Study Planner AI 🧠📅")

# Mode selection
mode = st.radio("Select Mode:", ["Interactive Chat Planner", "Syllabus Upload Planner"], horizontal=True)

if mode == "Interactive Chat Planner":
    st.subheader("Let's Build Your Perfect Study Plan 💬")
    
    # Clear history button
    if st.button("🧹 Clear Chat History"):
        st.session_state.messages = [
            SystemMessage(content="""You are an expert study planner assistant..."""),
            AIMessage(content="👋 Welcome! I'm your AI Study Assistant. Let's create your perfect learning plan! Could you first tell me what subject you want to focus on?")
        ]
        st.session_state.collected_data = {}
        st.session_state.plan_generated = False
        st.rerun()

    # Display chat history
    for msg in st.session_state.messages[1:]:  # Skip system message
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.write(msg.content)

    # Plan generation logic
    if st.session_state.plan_generated:
        plan_prompt = f"""Generate comprehensive study plan with:
        - Weekly schedule with time allocations
        - Topic-wise breakdown with key concepts
        - Curated resources from ResourceSearch tool
        - Practice exercises and projects
        - Progress tracking system
        
        Student Profile:
        {st.session_state.collected_data}
        
        Format for each topic:
        1. Topic Name
        2. Time Allocation
        3. Resources (use ResourceSearch if needed)
        4. Key Concepts
        5. Practice Exercises"""
        
        # Use .invoke() for final plan generation
        final_plan = groq_chat.invoke([
            HumanMessage(content=plan_prompt),
            SystemMessage(content="Use ResourceSearch tool for finding relevant resources")
        ])
        
        with st.expander("🌟 Your Personalized Study Plan", expanded=True):
            st.markdown(final_plan.content)
            st.download_button("📥 Download Plan", final_plan.content, file_name="study_plan.md")
        
        if st.button("🔄 Start New Plan"):
            st.session_state.clear()
            st.rerun()
    else:
        # Chat input
        if user_input := st.chat_input("Type your response..."):
            response = generate_response(user_input)
            st.rerun()

else:  # Syllabus Upload Planner
    st.subheader("Upload Your Syllabus 📄📷")
    
    # File upload section
    uploaded_file = st.file_uploader("Upload PDF", 
                                   type=["pdf"])
    
    syllabus_text = ""
    
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            
            if uploaded_file.type == "application/pdf":
                reader = PdfReader(tmp_file.name)
                for page in reader.pages:
                    syllabus_text += page.extract_text()
            
        
        st.success("✅ File processed successfully!")
    
    if not uploaded_file:
        syllabus_text = st.text_area("Or paste your syllabus text here:", height=200)
    
    col1, col2 = st.columns(2)
    with col1:
        total_days = st.number_input("Total available days:", min_value=1, value=30)
    with col2:
        daily_hours = st.number_input("Daily study hours:", min_value=1, value=2)
    
    if st.button("🚀 Generate from Syllabus"):
        if syllabus_text:
            syllabus_prompt = f"""Create detailed study plan from this syllabus:
            {syllabus_text}
            
            Requirements:
            - Duration: {total_days} days ({daily_hours} hrs/day)
            - Include topic-wise resource recommendations (use ResourceSearch tool)
            - Add practice tests schedule
            - Suggest reference books and online materials
            -Use the ResourceSearch tool only if additional resources (books, videos, online materials, etc.) are necessary.
            - Include project/work suggestions"""
            
            # Use .invoke() for syllabus-based plan generation
            response = groq_chat.invoke([
                HumanMessage(content=syllabus_prompt),
                SystemMessage(content="Use ResourceSearch tool for finding relevant resources")
            ])
            
            st.subheader("📚 Your Syllabus-Based Plan")
            st.markdown(response.content)
            st.download_button("📥 Download Plan", response.content, file_name="syllabus_plan.md")
        else:
            st.error("❌ Please upload/paste syllabus content")
