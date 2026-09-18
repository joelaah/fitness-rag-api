"""
Streamlit RAG Application
=========================
Interactive web interface for the Gemini + Qdrant RAG pipeline.
- File upload widget for ingesting documents into Qdrant Cloud.
- Chat input block for natural language queries.
- Uses @st.cache_resource to cache database and API client connections.
- Wires frontend to ingest.py, retrieve.py, and generate.py.
"""

from __future__ import annotations

import logging
import os
import cohere
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from google import genai
from qdrant_client import QdrantClient

from ingest import (
    COLLECTION_NAME,
    ensure_collection,
    ingest_texts,
    init_gemini_client,
    init_qdrant_client,
)
from retrieve import retrieve
from generate import generate_answer

# Load environment configuration
load_dotenv()

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cached Resource Initializers
# ---------------------------------------------------------------------------


@st.cache_resource
def get_qdrant_client() -> QdrantClient:
    """
    Establish and cache the Qdrant database connection.
    Decorated with @st.cache_resource so Streamlit reuses this singleton
    across reruns instead of reconnecting on every user interaction.
    """
    return init_qdrant_client()


@st.cache_resource
def get_gemini_client() -> genai.Client:
    """
    Initialize and cache the Google GenAI client instance.
    """
    return init_gemini_client()


@st.cache_resource
def get_cohere_client() -> cohere.ClientV2:
    """
    Initialize and cache the Cohere V2 client for reranking.
    """
    from retrieve import init_cohere_client
    return init_cohere_client()


# ---------------------------------------------------------------------------
# Helper: Text Chunking
# ---------------------------------------------------------------------------


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    """
    Simple text chunker by paragraphs or character size with overlap.
    """
    # First split by double newlines to respect paragraph boundaries
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []

    for para in paragraphs:
        if len(para) <= chunk_size:
            chunks.append(para)
        else:
            # Sub-split long paragraphs
            start = 0
            while start < len(para):
                end = start + chunk_size
                chunks.append(para[start:end].strip())
                start += chunk_size - chunk_overlap

    return [c for c in chunks if c]


# ---------------------------------------------------------------------------
# Streamlit UI Configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Gemini + Qdrant RAG Assistant",
    page_icon="🧠",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Interactive 3D WebGL Neural Core (Three.js)
# ---------------------------------------------------------------------------
THREE_JS_SCENE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    background: transparent;
    overflow: hidden;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    user-select: none;
  }
  #scene-card {
    position: relative;
    width: 100%;
    height: 275px;
    border-radius: 20px;
    overflow: hidden;
    background: radial-gradient(circle at 50% 50%, rgba(17, 24, 39, 0.7) 0%, rgba(10, 14, 23, 0.95) 100%);
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 0 16px 40px -10px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.15);
  }
  #webgl-canvas {
    width: 100%;
    height: 100%;
    display: block;
    cursor: grab;
  }
  #webgl-canvas:active {
    cursor: grabbing;
  }
  .hud-badge {
    position: absolute;
    top: 16px;
    left: 20px;
    pointer-events: none;
    display: flex;
    flex-direction: column;
    gap: 4px;
    z-index: 10;
  }
  .hud-title-row {
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .hud-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #10B981;
    box-shadow: 0 0 10px #10B981, 0 0 20px rgba(16, 185, 129, 0.6);
    animation: blink 2s infinite ease-in-out;
  }
  @keyframes blink {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.35; transform: scale(0.8); }
  }
  .hud-title {
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #F8FAFC;
    text-shadow: 0 0 12px rgba(99, 102, 241, 0.6);
  }
  .hud-subtitle {
    font-size: 11px;
    color: #94A3B8;
    letter-spacing: 0.04em;
  }
  .hud-stats-row {
    position: absolute;
    bottom: 14px;
    right: 18px;
    display: flex;
    gap: 8px;
    pointer-events: none;
    flex-wrap: wrap;
    z-index: 10;
  }
  .stat-pill {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.12);
    padding: 4px 11px;
    border-radius: 14px;
    font-size: 11px;
    color: #CBD5E1;
    letter-spacing: 0.03em;
    display: flex;
    align-items: center;
    gap: 5px;
  }
  .stat-pill-accent {
    border-color: rgba(99, 102, 241, 0.4);
    background: rgba(99, 102, 241, 0.15);
    color: #C7D2FE;
  }
  .stat-pill-cyan {
    border-color: rgba(6, 182, 212, 0.4);
    background: rgba(6, 182, 212, 0.12);
    color: #A5F3FC;
  }
</style>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
</head>
<body>
<div id="scene-card">
  <canvas id="webgl-canvas"></canvas>
  <div class="hud-badge">
    <div class="hud-title-row">
      <div class="hud-dot"></div>
      <div class="hud-title">3D Neural Vector Core</div>
    </div>
    <div class="hud-subtitle">Qdrant Topology • text-embedding-004 (768-D) • Cohere Rerank</div>
  </div>
  <div class="hud-stats-row">
    <div class="stat-pill stat-pill-accent">⚡ Gemini Core</div>
    <div class="stat-pill stat-pill-cyan">🎯 2-Stage Retrieval</div>
    <div class="stat-pill">🔄 Drag 3D Model</div>
  </div>
</div>

<script>
  const canvas = document.getElementById('webgl-canvas');
  const container = document.getElementById('scene-card');
  
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 1000);
  camera.position.z = 6.2;

  const renderer = new THREE.WebGLRenderer({ canvas: canvas, alpha: true, antialias: true });
  renderer.setSize(container.clientWidth, container.clientHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

  // Ambient & Directional Lights
  const ambientLight = new THREE.AmbientLight(0x6366f1, 0.9);
  scene.add(ambientLight);

  const dirLight1 = new THREE.DirectionalLight(0x38bdf8, 1.4);
  dirLight1.position.set(5, 5, 5);
  scene.add(dirLight1);

  const dirLight2 = new THREE.DirectionalLight(0xa855f7, 1.3);
  dirLight2.position.set(-5, -5, -2);
  scene.add(dirLight2);

  // Group to rotate everything
  const coreGroup = new THREE.Group();
  scene.add(coreGroup);

  // 1. Central Core Sphere (Glowing Inner Pulsing Core)
  const coreGeo = new THREE.SphereGeometry(0.9, 32, 32);
  const coreMat = new THREE.MeshPhongMaterial({
    color: 0x4f46e5,
    emissive: 0x312e81,
    specular: 0x818cf8,
    shininess: 95,
    transparent: true,
    opacity: 0.88
  });
  const coreSphere = new THREE.Mesh(coreGeo, coreMat);
  coreGroup.add(coreSphere);

  // 2. Wireframe Icosahedron (Neural Lattice Outer Hull)
  const icoGeo = new THREE.IcosahedronGeometry(1.55, 2);
  const icoMat = new THREE.MeshBasicMaterial({
    color: 0x38bdf8,
    wireframe: true,
    transparent: true,
    opacity: 0.38
  });
  const icoMesh = new THREE.Mesh(icoGeo, icoMat);
  coreGroup.add(icoMesh);

  // 3. Node Vertices (Glowing Synapses)
  const icoVertices = icoGeo.attributes.position.array;
  const nodesGeo = new THREE.BufferGeometry();
  nodesGeo.setAttribute('position', new THREE.BufferAttribute(icoVertices, 3));
  const nodesMat = new THREE.PointsMaterial({
    color: 0xa5b4fc,
    size: 0.085,
    transparent: true,
    opacity: 0.95
  });
  const nodesMesh = new THREE.Points(nodesGeo, nodesMat);
  coreGroup.add(nodesMesh);

  // 4. Orbiting Gyroscope Rings
  const ringGeo1 = new THREE.TorusGeometry(2.05, 0.02, 16, 100);
  const ringMat1 = new THREE.MeshBasicMaterial({ color: 0x818cf8, transparent: true, opacity: 0.65 });
  const ring1 = new THREE.Mesh(ringGeo1, ringMat1);
  ring1.rotation.x = Math.PI / 3;
  coreGroup.add(ring1);

  const ringGeo2 = new THREE.TorusGeometry(2.28, 0.018, 16, 100);
  const ringMat2 = new THREE.MeshBasicMaterial({ color: 0x06b6d4, transparent: true, opacity: 0.6 });
  const ring2 = new THREE.Mesh(ringGeo2, ringMat2);
  ring2.rotation.y = Math.PI / 4;
  ring2.rotation.x = -Math.PI / 6;
  coreGroup.add(ring2);

  // 5. Surrounding Vector Embeddings Particle Cloud (240 particles)
  const particleCount = 240;
  const particleGeo = new THREE.BufferGeometry();
  const particlePositions = new Float32Array(particleCount * 3);
  for (let i = 0; i < particleCount * 3; i += 3) {
    const r = 2.2 + Math.random() * 2.3;
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.acos((Math.random() * 2) - 1);
    particlePositions[i] = r * Math.sin(phi) * Math.cos(theta);
    particlePositions[i + 1] = r * Math.sin(phi) * Math.sin(theta);
    particlePositions[i + 2] = r * Math.cos(phi);
  }
  particleGeo.setAttribute('position', new THREE.BufferAttribute(particlePositions, 3));
  const particleMat = new THREE.PointsMaterial({
    color: 0x38bdf8,
    size: 0.045,
    transparent: true,
    opacity: 0.8
  });
  const particleMesh = new THREE.Points(particleGeo, particleMat);
  coreGroup.add(particleMesh);

  // Mouse Interaction (Drag and Tilt)
  let isDragging = false;
  let prevMouseX = 0;
  let prevMouseY = 0;
  let targetRotX = 0;
  let targetRotY = 0;

  window.addEventListener('mousedown', (e) => {
    isDragging = true;
    prevMouseX = e.clientX;
    prevMouseY = e.clientY;
  });

  window.addEventListener('mouseup', () => { isDragging = false; });

  window.addEventListener('mousemove', (e) => {
    const rect = container.getBoundingClientRect();
    const relX = (e.clientX - rect.left) / rect.width - 0.5;
    const relY = (e.clientY - rect.top) / rect.height - 0.5;

    if (isDragging) {
      const deltaX = e.clientX - prevMouseX;
      const deltaY = e.clientY - prevMouseY;
      coreGroup.rotation.y += deltaX * 0.008;
      coreGroup.rotation.x += deltaY * 0.008;
      prevMouseX = e.clientX;
      prevMouseY = e.clientY;
    } else {
      targetRotY = relX * 0.5;
      targetRotX = relY * 0.5;
    }
  });

  // Animation Loop
  let clock = new THREE.Clock();
  function animate() {
    requestAnimationFrame(animate);
    const elapsedTime = clock.getElapsedTime();

    if (!isDragging) {
      coreGroup.rotation.y += 0.006;
      coreGroup.rotation.x += (targetRotX - coreGroup.rotation.x) * 0.04;
    }

    ring1.rotation.z += 0.009;
    ring2.rotation.z -= 0.007;

    const pulse = 1 + Math.sin(elapsedTime * 2.5) * 0.035;
    coreSphere.scale.set(pulse, pulse, pulse);
    icoMesh.rotation.y -= 0.0025;

    renderer.render(scene, camera);
  }
  animate();

  window.addEventListener('resize', () => {
    if (!container) return;
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
  });
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Glassmorphism & Modern Aesthetic Design System
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global Typography & Text */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #E2E8F0;
    }

    /* Ambient Space Atmosphere */
    .stApp {
        background: radial-gradient(circle at 12% 12%, rgba(99, 102, 241, 0.16) 0%, transparent 45%),
                    radial-gradient(circle at 88% 16%, rgba(168, 85, 247, 0.13) 0%, transparent 48%),
                    radial-gradient(circle at 50% 92%, rgba(6, 182, 212, 0.10) 0%, transparent 55%),
                    #0A0E17 !important;
        background-attachment: fixed;
    }

    /* Force dark background on bottom containers to eradicate white bars */
    footer {
        visibility: hidden !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    [data-testid="stBottom"],
    [data-testid="stBottomBlockContainer"],
    .stBottom,
    div[data-testid="stBottomBlockContainer"] > div,
    div:has(> [data-testid="stChatInput"]),
    div:has(> div[data-testid="stChatInput"]) {
        background: transparent !important;
        background-color: transparent !important;
    }

    /* Main container layout padding */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 5rem !important;
        max-width: 1260px !important;
    }

    /* Top Navigation Bar */
    header[data-testid="stHeader"] {
        background: rgba(10, 14, 23, 0.45) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05) !important;
    }

    /* Translucent Sidebar */
    section[data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.72) !important;
        backdrop-filter: blur(24px) !important;
        -webkit-backdrop-filter: blur(24px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        font-family: 'Outfit', sans-serif !important;
        color: #F8FAFC !important;
        letter-spacing: -0.02em;
    }

    /* File Uploader Dropzone */
    [data-testid="stFileUploadDropzone"],
    section[data-testid="stFileUploadDropzone"],
    div[data-testid="stFileUploadDropzone"] {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1.5px dashed rgba(99, 102, 241, 0.35) !important;
        border-radius: 16px !important;
        padding: 1.25rem 1rem !important;
        backdrop-filter: blur(12px) !important;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }

    [data-testid="stFileUploadDropzone"]:hover {
        border-color: rgba(168, 85, 247, 0.7) !important;
        background: rgba(99, 102, 241, 0.08) !important;
        box-shadow: 0 0 20px rgba(99, 102, 241, 0.2) !important;
    }

    [data-testid="stFileUploadDropzone"] * {
        color: #CBD5E1 !important;
    }

    [data-testid="stFileUploadDropzone"] button {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.3) 0%, rgba(168, 85, 247, 0.25) 100%) !important;
        border: 1px solid rgba(255, 255, 255, 0.18) !important;
        color: #FFFFFF !important;
        border-radius: 10px !important;
        font-weight: 500 !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25) !important;
        transition: all 0.2s ease !important;
    }

    [data-testid="stFileUploadDropzone"] button:hover {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.5) 0%, rgba(168, 85, 247, 0.45) 100%) !important;
        border-color: rgba(255, 255, 255, 0.35) !important;
    }

    /* Glass Cards for Metrics */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.04) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 16px !important;
        padding: 14px 18px !important;
        backdrop-filter: blur(14px) !important;
        box-shadow: 0 6px 24px rgba(0, 0, 0, 0.3) !important;
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    }

    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        border-color: rgba(99, 102, 241, 0.5) !important;
        box-shadow: 0 8px 28px rgba(99, 102, 241, 0.2) !important;
    }

    /* Buttons with Neon Violet/Indigo Gradient */
    .stButton > button {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 12px !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
        padding: 0.6rem 1.4rem !important;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
        box-shadow: 0 4px 18px rgba(79, 70, 229, 0.4) !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 28px rgba(124, 58, 237, 0.65) !important;
        border-color: rgba(255, 255, 255, 0.4) !important;
    }

    .stButton > button:active {
        transform: translateY(0px) !important;
    }

    /* Frosted Chat Bubbles */
    div[data-testid="stChatMessage"] {
        background: rgba(15, 23, 42, 0.65) !important;
        border: 1px solid rgba(255, 255, 255, 0.09) !important;
        border-radius: 20px !important;
        padding: 1.35rem 1.6rem !important;
        margin-bottom: 1.1rem !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3) !important;
    }

    div[data-testid="stChatMessage"]:hover {
        border-color: rgba(255, 255, 255, 0.16) !important;
    }

    /* User Message Bubble Glow */
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageUser"]) {
        background: linear-gradient(135deg, rgba(79, 70, 229, 0.18) 0%, rgba(124, 58, 237, 0.12) 100%) !important;
        border: 1px solid rgba(99, 102, 241, 0.35) !important;
        box-shadow: 0 8px 28px rgba(99, 102, 241, 0.15) !important;
    }

    /* Assistant Message Accent Line */
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAssistant"]) {
        border-left: 3.5px solid #6366F1 !important;
    }

    /* Floating Chat Input Bar */
    div[data-testid="stChatInput"] {
        border-radius: 22px !important;
        background: rgba(15, 23, 42, 0.85) !important;
        backdrop-filter: blur(24px) !important;
        -webkit-backdrop-filter: blur(24px) !important;
        border: 1px solid rgba(99, 102, 241, 0.35) !important;
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.45) !important;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }

    div[data-testid="stChatInput"]:focus-within {
        border-color: rgba(99, 102, 241, 0.85) !important;
        box-shadow: 0 0 30px rgba(99, 102, 241, 0.4) !important;
    }

    /* Expanders & Accordions */
    div[data-testid="stExpander"] {
        background: rgba(255, 255, 255, 0.025) !important;
        border: 1px solid rgba(255, 255, 255, 0.09) !important;
        border-radius: 14px !important;
    }

    /* Custom Header Hero */
    .hero-container {
        padding: 0.2rem 0 0.8rem 0;
        margin-bottom: 0.75rem;
    }

    .hero-title {
        font-family: 'Outfit', sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #FFFFFF 20%, #C7D2FE 60%, #818CF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 0.4rem 0;
    }

    .hero-badges {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        flex-wrap: wrap;
        margin-top: 0.4rem;
    }

    .badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 500;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: #CBD5E1;
        backdrop-filter: blur(8px);
    }

    .badge-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #10B981;
        box-shadow: 0 0 8px #10B981;
    }

    /* Showcase Cards */
    .showcase-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 1rem;
        margin: 1.25rem 0;
    }
    .showcase-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.2rem;
        backdrop-filter: blur(14px);
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
    }
    .showcase-card:hover {
        transform: translateY(-3px);
        border-color: rgba(99, 102, 241, 0.45);
        background: rgba(99, 102, 241, 0.06);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
    }
    .showcase-icon {
        font-size: 1.5rem;
        margin-bottom: 0.5rem;
    }
    .showcase-title {
        font-family: 'Outfit', sans-serif;
        font-size: 0.95rem;
        font-weight: 600;
        color: #F8FAFC;
        margin-bottom: 0.3rem;
    }
    .showcase-desc {
        font-size: 0.8rem;
        color: #94A3B8;
        line-height: 1.45;
    }
    </style>

    <div class="hero-container">
        <h1 class="hero-title">🧠 Gemini + Qdrant RAG Assistant</h1>
        <div class="hero-badges">
            <span class="badge"><span class="badge-dot"></span> System Online</span>
            <span class="badge">gemini-embedding-2</span>
            <span class="badge">Qdrant Cloud Vectors</span>
            <span class="badge">Cohere Rerank v2</span>
            <span class="badge">gemini-3.6-flash</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Render Interactive 3D WebGL Neural Core
components.html(THREE_JS_SCENE_HTML, height=285)


# ---------------------------------------------------------------------------
# Role-Based Access: Admin vs User
# ---------------------------------------------------------------------------

def _get_admin_password() -> str:
    """Read admin password from st.secrets (deployed) or .env (local)."""
    try:
        return st.secrets["ADMIN_PASSWORD"]
    except (KeyError, FileNotFoundError):
        return os.environ.get("ADMIN_PASSWORD", "")


if "role" not in st.session_state:
    st.session_state["role"] = "user"

# ---------------------------------------------------------------------------
# Sidebar: Role-Aware
# ---------------------------------------------------------------------------

with st.sidebar:
    # ── Admin Panel (visible only to authenticated admins) ──
    if st.session_state["role"] == "admin":
        st.header("📄 Document Ingestion")

        uploaded_files = st.file_uploader(
            "Upload documents (.txt, .md, .pdf)",
            type=["txt", "md", "pdf"],
            accept_multiple_files=True,
            help="Upload text or PDF files to embed and index into your Qdrant vector database.",
        )

        chunk_size = st.slider("Chunk Size (characters)", min_value=200, max_value=1500, value=500, step=100)

        if st.button("🚀 Ingest Documents", use_container_width=True):
            if not uploaded_files:
                st.warning("Please select at least one file to upload.")
            else:
                try:
                    with st.spinner("Connecting to clients & initializing collection..."):
                        gemini = get_gemini_client()
                        qdrant = get_qdrant_client()
                        ensure_collection(qdrant)

                    total_chunks = 0
                    all_chunks: list[str] = []

                    for file in uploaded_files:
                        if file.name.lower().endswith(".pdf"):
                            import io
                            import pypdf
                            reader = pypdf.PdfReader(io.BytesIO(file.read()))
                            content = "\n\n".join(page.extract_text() or "" for page in reader.pages)
                        else:
                            content = file.read().decode("utf-8", errors="replace")
                        file_chunks = chunk_text(content, chunk_size=chunk_size)
                        all_chunks.extend(file_chunks)

                    if not all_chunks:
                        st.warning("No readable text found in uploaded files.")
                    else:
                        with st.spinner(f"Embedding and upserting {len(all_chunks)} chunks..."):
                            # Combine all filenames for the document_name metadata
                            doc_names = ", ".join(f.name for f in uploaded_files)
                            upserted_count = ingest_texts(
                                gemini, qdrant, all_chunks,
                                document_name=doc_names,
                            )
                            st.success(f"✅ Ingested {upserted_count} chunks into '{COLLECTION_NAME}'")

                except Exception as e:
                    st.error(f"❌ Ingestion failed: {e}")

        st.divider()

        st.subheader("⚙️ System Status")
        try:
            qdrant = get_qdrant_client()
            if qdrant.collection_exists(COLLECTION_NAME):
                col_info = qdrant.get_collection(COLLECTION_NAME)
                st.success(f"Connected to Qdrant: `{COLLECTION_NAME}`")
                st.metric("Indexed Vectors", col_info.points_count or 0)
            else:
                st.info(f"Collection `{COLLECTION_NAME}` does not exist yet. Upload files to create it.")
        except Exception as e:
            st.warning(f"Database connection: {e}")

        st.divider()

        if st.button("🔓 Logout (Admin)", use_container_width=True):
            st.session_state["role"] = "user"
            st.rerun()

    # ── User Panel (visible to everyone) ──
    else:
        st.markdown(
            """
            <div style="text-align:center; padding: 1.5rem 0.5rem;">
                <div style="font-size: 2rem; margin-bottom: 0.3rem;">🧠</div>
                <div style="font-family: 'Outfit', sans-serif; font-size: 1rem; font-weight: 600; color: #F8FAFC; margin-bottom: 0.2rem;">RAG Assistant</div>
                <div style="font-size: 0.78rem; color: #94A3B8;">Ask questions about your indexed documents</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        with st.expander("🔑 Admin Login"):
            admin_pw = st.text_input("Enter admin password", type="password", key="admin_pw_input")
            if st.button("Login", use_container_width=True):
                expected = _get_admin_password()
                if admin_pw and admin_pw == expected:
                    st.session_state["role"] = "admin"
                    st.rerun()
                else:
                    st.error("Incorrect password.")

    # ── Always visible ──
    st.divider()
    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()




# ---------------------------------------------------------------------------
# Main Panel: Chat & Retrieval Interface
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Empty State Showcase Cards
if not st.session_state["messages"]:
    st.markdown(
        """
        <div class="showcase-grid">
            <div class="showcase-card">
                <div class="showcase-icon">⚡</div>
                <div class="showcase-title">Fast Semantic Search</div>
                <div class="showcase-desc">High-dimensional vector embeddings powered by Google Gemini and indexed in Qdrant Cloud.</div>
            </div>
            <div class="showcase-card">
                <div class="showcase-icon">🎯</div>
                <div class="showcase-title">Two-Stage Cohere Rerank</div>
                <div class="showcase-desc">Initial top-25 vector candidates pruned to the 3 most precise chunks via Cohere rerank v2.</div>
            </div>
            <div class="showcase-card">
                <div class="showcase-icon">🛡️</div>
                <div class="showcase-title">Grounded Synthesis</div>
                <div class="showcase-desc">Gemini synthesizes factual responses with strict citations from your uploaded materials.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Display previous conversation messages
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "context" in msg and msg["context"]:
            with st.expander("🔍 View Retrieved Context"):
                st.text(msg["context"])

# Chat input widget
user_query = st.chat_input("Ask a question based on your indexed documents...")

if user_query:
    # Append & display user message
    st.session_state["messages"].append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Process query
    with st.chat_message("assistant"):
        try:
            gemini = get_gemini_client()
            qdrant = get_qdrant_client()

            with st.spinner("Retrieving & reranking context (25 → top 3)..."):
                co = get_cohere_client()
                retrieved_context = retrieve(
                    query=user_query,
                    gemini=gemini,
                    qdrant=qdrant,
                    co=co,
                )

            with st.spinner("Generating grounded answer with Gemini..."):
                answer = generate_answer(
                    query=user_query,
                    context=retrieved_context,
                    gemini=gemini,
                )

            st.markdown(answer)

            if retrieved_context:
                with st.expander("🔍 View Reranked Context (Top 3 Results)"):
                    st.text(retrieved_context)
            else:
                st.caption("ℹ️ No relevant context was retrieved.")

            # Record in session state
            st.session_state["messages"].append({
                "role": "assistant",
                "content": answer,
                "context": retrieved_context,
            })

        except Exception as err:
            error_msg = f"An error occurred: {err}"
            st.error(error_msg)
            st.session_state["messages"].append({
                "role": "assistant",
                "content": error_msg,
            })
