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

