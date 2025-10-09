## StoryPlus
HACK CMU PROJECT: AI Storytelling Platform
An AI-powered storytelling app that generates infinitely long stories in real-time, creating a TikTok-style feed of narratives.

Overview: 
- RUN112 is a 2.5D tunnel runner game where players navigate through a procedurally generated tunnel, avoiding the holes and making sure the player stays on the platforms.

Key Features: 
- AI Story Generation:
    - Fine-tuned LLaMa 3.1 by training it on 100k+ story dataset
    - Real-time streaming generation with Server-Sent Events(SSE)
    - Stories can go on indefinitely by passing summaries between generations while maintaining consistency and focus on each story
    - Deduplication using SimHash algorithm to prevent repetetive content
- Text-to-Speech Integration:
    - OpenAI TTS API with 8 different voices and synchronized playback 
- Custom Physics Engine: 
    - Gravity Simulation with vertical velocity and acceleration
    - Collision detection algorithms
    - Platform rotation mechanics
    - Smooth character movement
- Recommendation System:
    - Sentence-transformers embeddings and cosine similarity matching for content recommendations
    - Automatic keyword extraction from story content

How To Play:
- Controls:
    - Left/Right Arrow Keys: Move character horizontally
    - Spacebar: Jump
    - Mouse: Navigate menus and pause
- Gameplay:
    - Start by clicking "PLAY" on the homescreen
    - Navigate through the tunnel by jumping and moving left and right, avoiding the black holes that come towards the screen
    - Jump on other platforms to rotate the tunnel and avoid the black holes
 
Project Structure:
- page.tsx: 

Installation Requirements: 
- Python 3.8+
- pip package manager
- cmu-graphics and pygame installation

Team Contributions:
- Genti Aliaj:
    - Ball movement physics and gravity simulation
    - 3D tunnel rendering and mathematical transformations
    - Platform rotation mechanics and collision detection
    - Procedural generation and random hole spawning
- Eileen Jung:
    - Designed home, settings, and game over screen
    - Implemented level progression system
    - Integrated music
- Gabbie Boone:
    - Color customization system
    - Pause functionality
    - Character design

We made this game for a term project for **15-112: Fundamentals of Programming and Computer Science**.
  
