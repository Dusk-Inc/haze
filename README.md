![alt text](resources/Haze%20Readme%20Banner.png)

Haze is a continuously learning, self-organizing neural mesh designed to evolve across multiple contexts, data sources, and problem domains. Unlike traditional neural networks with static layers and fixed architectures, Haze is built from modular components that dynamically grow, adapt, and connect in flexible, directional ways.

## Project Status
**This is a research preview.**
\
Haze is under active development and not yet production-ready. Core features like real-time learning, modular interpreters, and signal propagation are functional, but predictive capabilities are still evolving.

## Getting Started
Haze is not yet available as an installable library. To get started, clone the repository:
```
git clone https://github.com/D-U-S-K-D-E-V/haze.git
```
### Containers
If you're using Docker for a dev container, consult your IDE’s documentation. The included container handles all required setup automatically.

### Installation
If you're not using Docker, create a virtual environment and install dependencies:
```
python -m venv .env
source .env/bin/activate     # On Windows: .env\Scripts\activate
pip install -r requirements.txt
```

Run a test case using:
```
python main.py
```

### Requirements
- Python: Tested with version 3.13.3

### Optional: Docker
While not required, Docker is recommended. It ensures consistent setup and simplifies debugging for contributors.


## Overview
Haze is a modular, self-organizing neural system inspired by biological cognition. Instead of layers, it uses three loosely structured regions of neurons connected by directional, weighted links. The system evolves through experience rather than offline retraining.

### What problem does Haze solve?
Haze addresses the rigidity of traditional deep learning architectures. While agentic systems show promise, they are often computationally expensive. Haze introduces a lightweight, evolving architecture with adaptive encoders/decoders and a flexible mesh, aiming to create a single, general-purpose model capable of answering diverse business and scientific questions.


### What makes it different from other neural network libraries or generative models?
#### Flexible Architecture
Haze removes the strict 1:1 mapping between input features and weights. Connections act like myelin in biological systems, determining energy decay through a network of sparse, directional connections. The network can grow or prune neurons dynamically to adapt to new tasks or data.
#### Signal Aggregation
Signals can accumulate over time before triggering a prediction. This lets Haze "form an opinion", e.g., aggregating an hour’s worth of crypto data before making a decision, or later including sentiment from text as additional signal.
#### Modular I/O Plugins
Encoders transform input into normalized vectors and grow sensor neurons as needed. Decoders map signal into actionable output using motor neurons, which can be activated/deactivated dynamically. Both are designed to support extensibility across data types.
#### Multi-modal Thinking
Haze supports experimental "Idea Chains". These are sequential encoder-decoder paths that mimic reasoning. This allows multiple modalities (text, vectors, images) to be processed together, building richer context.
\
*This is an experimental feature as its strategic value has not been tested as of yet.*


## Architecture Summary
### Neurons
Haze utilizes 3 different "types" of Neurons.
- Sensor Neurons: Accept input from encoders and initiate signal flow.
- Interneurons: Form the conceptual mesh that learns internal associations.
- Motor Neurons: Produce output by averaging signals and outputs the aggregate signal to decoders when called.

### Connections
Connections are sparse, directional, and weighted. They decay signals unless strengthened by learning. The mesh evolves by growing new connections when the system underperforms.

### Encoders
Encoders normalize input into signals and create new sensor neurons as the input expands. The current version includes a numeric encoder; future plans include text, image, and sound encoders.

### Decoders
Decoders interpret motor neuron signals into output. Available decoder types include:
- Argmax
- Softmax
- Regression
- TopK
- BitMask

### Layers
These soft layers guide signal flow:
- Aperture: Richly connected to sensor neurons. Allows all features to influence downstream signals.
- Nexus: Sparse mesh where conceptual processing occurs.
- Terminus: Funnels signals to motor neurons and decoders.

### Signal Propagation
- Signals are values between 0.1 and 0.9 with an associated feature ID.
- Propagation involves multiplying signals by connection weights.
- Signals decay naturally unless reinforced.
- A geometric mean is used to stabilize signal strength across long paths.
- Looping signals are blocked to avoid recursion.

### Learning
Instead of backpropagation, Haze uses a forward-only learning mechanism:
- During propagation, used connections are logged.
- After output, connections are rewarded or penalized based on:
- Accuracy of prediction
- Confidence of motor neuron activations
- This process is vectorized and lightweight.

## Current Features
- Self-Organizing Mesh: Grows/prunes neurons and connections automatically.
- Minimal Encoder/Decoder Support: Accepts float arrays and processes outputs.
- Recovery Logic: Strengthens unused connections if signals fail to reach motors.
- Sequential Output: Feeds its own outputs back for multi-step predictions.
- Chaining System: Experimental support for idea-to-idea chains.
- Opinion Formation: Allows signal accumulation before predictions.
- Save/Load: Supports saving state with `persist=True`.

## Future Plans
Haze is currently a very early research preview, meant to spark discussion and ideation around the project's future. However, there are some known milestones to overcome in the future including:

### Vectorized Network
Currently, the network operates as a simulation, with explicit connections and neurons. However, this process, even when parallelized the way it is currently, is too slow for a production system. Development is already underway on a version of Haze in which the network is represented and managed as a matrix that signal propagates through. This will allow for vectorized propagation and make Haze significantly faster comparatively.


### Expanded I/O Capabilities
Currently, Haze has minimal encoders/decoders for conceptual development and testing. However, for Haze to be production ready, it must handle more than just arrays of floats. New encoders that can process images, sound, text data, unstructured data, and more will be essential for Haze's development.

### Automated Chaining
Allow Haze to choose its own chain length and modality flow based on context.

### Cloud Scaling
Haze’s sparse mesh is well-suited for cross-server scaling similar to microservices.

### PIP Deployment
We plan to make Haze installable via pip once the codebase stabilizes.


## Feedback and Contribution
This project is in its infancy and would benefit greatly from anyone interested in improving the AI ecosystem. Feel free to: 
- [submit and issue](https://github.com/D-U-S-K-D-E-V/haze/issues)
- [start a discussion](https://github.com/D-U-S-K-D-E-V/haze/discussions)
- join our [Discord](https://discord.gg/hM62aYgJ) community!

We are currently working on our code style guide, and will attach it as soon as it's available.


## License
Apache 2.0


## Contact
Have questions or want to collaborate?

📧 questions@dusk-inc.com
\
🌐 [dusk-inc.com](https://dusk-inc.com/)
\
[![Linkedin](https://i.sstatic.net/gVE0j.png) LinkedIn](https://www.linkedin.com/company/106766346)