# MontLock

MontLock is an innovative security solution for Windows computers that uses biometric mouse movement patterns to authenticate users. The system monitors mouse movements in real-time and compares them against a trained model of the authorized user's behavior. If unauthorized use is detected, the cursor is immobilized for 60 seconds.

## Features

- **Biometric Authentication**: Uses unique mouse movement patterns for user verification
- **Real-time Monitoring**: Continuously analyzes mouse behavior
- **Automatic Protection**: Locks cursor when unauthorized use is detected
- **User Training**: Creates personalized biometric profiles

## Biometric Parameters

MontLock analyzes several aspects of mouse movement:
- Mouse speed (average, variations, patterns)
- Reaction time to stimuli
- Movement accuracy and precision
- Trajectory patterns and curves
- Acceleration and jerk profiles

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/MontLock.git
cd MontLock

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training the Model

Before using MontLock, you need to train it with your mouse movement patterns:

```bash
python main.py train
```

Follow the on-screen instructions to complete the training process.

### Activating Protection

To start MontLock protection:

```bash
python main.py protect
```

The system will run in the background, monitoring mouse movements and securing your computer.

### Stopping Protection

To stop MontLock:

```bash
python main.py stop
```

## System Requirements

- Windows 10 or later
- Python 3.8 or later
- Administrator privileges (for cursor control)

## License

MIT 