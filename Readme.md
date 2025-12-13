# RadioSport STEM Handbook

## Advanced STEM Education Program

Self-Sustaining Offline Technology & Non-Commercial Electronic Communication Education

Empowering Youth Through Amateur Radio, AI, Coding, and Field Science

🔬 SCIENCE | 💻 TECHNOLOGY | ⚙️ ENGINEERING | 📐 MATHEMATICS

**Version 2.0**

---

## 📋 Table of Contents

- [1. Mission Statement & STEM Excellence](#1-mission-statement--stem-excellence)
- [2. Program Structure](#2-program-structure)
- [3. Solar & Battery Power Systems](#3-solar--battery-power-systems)
- [4. HF/VHF/UHF Communications](#4-hfvhfuhf-communications)
- [5. Artificial Intelligence](#5-artificial-intelligence)
- [6. Advanced Coding: C++ & Python](#6-advanced-coding-c--python)
- [7. Hands-On Projects](#7-hands-on-projects)
- [8. Field Operations](#8-field-operations)
- [9. Awards System](#9-awards-system)

---

## 1. Mission Statement & STEM Excellence

### 🎯 Our Mission
**RadioSport develops advanced offline youth competence in cutting-edge technologies that bridge theoretical knowledge with real-world applications.**

Focuses on self-reliance in disconnected environments, building innovation and problem-solving for STEM careers.

### Core Competencies
- 🤖 **Artificial Intelligence**: Offline algorithms for optimization.
- 📻 **Radio Communication**: HF/VHF/UHF for connectivity.
- 💻 **Software Development**: C++/Python for signal processing.
- ⚡ **Renewable Energy**: Solar/battery for sustainable power.

### Why RadioSport is Exceptional STEM
- **Physics**: Observe wave propagation.
- **Mathematics**: Calculate wavelengths and designs.
- **Engineering**: Build and test systems.
- **Computer Science**: AI for signal analysis.
- **Environmental Science**: Sustainable energy systems.

### Program Duration & Structure

| Component       | Details                              |
|-----------------|--------------------------------------|
| Duration        | 9 months per cohort                  |
| Weekly Sessions | 3 hours (theory + coding + hands-on)|
| Monthly Field Days | 8 hours outdoor deployment        |

Progressive skill-building from fundamentals to capstone projects.

---

## 2. Program Structure

### Knowledge Streams

#### Offline AI & Computational Thinking
- Algorithms: Search, heuristics, constraints.
- ML: K-means, decision trees, naïve Bayes.
- Applications: Radio optimization, message routing.

#### Radio & RF Theory
- Fundamentals: Frequency, wavelength, impedance.
- Antennas: Dipoles, verticals, Yagis.
- Propagation: VHF/UHF line-of-sight, HF ionospheric.
- Modes: AFSK, D-STAR, PSK31.
- SDR: Signal analysis.

#### Coding & Software Engineering
- Python: Prototyping, data analysis.
- C++: High-performance processing.
- Tools: Propagation simulators, demodulation.

### Participant Requirements
- Minimum 80% attendance
- Weekly experiment logs
- Team collaboration
- Safe equipment operation

### Specialization Tracks

| Track                | Focus Areas                              |
|----------------------|------------------------------------------|
| RF Engineering       | Antenna design, propagation              |
| AI Signal Intelligence | ML, classification, optimization       |
| Communications       | Operations, digital modes, networks     |
| Software Development | DSP, SDR apps, automation               |

---

## 3. Solar & Battery Power Systems

Enable off-grid operations, teaching sustainable engineering.

### Solar Power Fundamentals
**Components**: Panel → Controller → Battery → Equipment
- 100W panel: ~400Wh in 4 sun hours.
- MPPT controller: Optimizes harvest.
- 35Ah LiFePO4: Night storage.
- Connectors: Anderson Powerpole.

#### Key Concepts
- Photovoltaic effect: Sunlight to current.
- Peak hours: 1000 W/m² equivalent.
- Efficiency: Monocrystalline 18-22%.
- MPPT: Tracks max power point.

### Battery Technology

| Type      | Voltage | Energy Density | Lifespan      | Best Use          |
|-----------|---------|----------------|---------------|-------------------|
| Lead Acid | 12V     | 35-40 Wh/kg    | 300-500 cycles| Budget stations   |
| LiFePO4   | 12.8V   | 90-120 Wh/kg   | 2000-5000 cycles| Portable ops     |
| Li-Ion    | 14.8V   | 150-200 Wh/kg  | 500-1000 cycles| Handhelds        |

### Power Budget Example: 8-Hour Deployment

| Equipment          | Power | Duration | Energy |
|--------------------|-------|----------|--------|
| VHF Radio (TX)     | 50W   | 1h       | 50 Wh  |
| VHF Radio (RX)     | 2W    | 7h       | 14 Wh  |
| Pi + SDR           | 15W   | 8h       | 120 Wh |
| Laptop             | 45W   | 4h       | 180 Wh |
| **Total**          |       |          | **364 Wh** |

Requirements: 35Ah battery, 100W panel, 10A MPPT.

### Exercises
- Panel: Measure I-V curve, fill factor.
- Battery: Discharge testing, capacity calc.

---

## 4. HF/VHF/UHF Communications

### Band Characteristics

| Band         | Frequency    | Propagation     | Range        |
|--------------|--------------|-----------------|--------------|
| HF (20m)     | 14-14.35 MHz | Ionospheric     | Worldwide   |
| HF (40m)     | 7-7.3 MHz    | Skywave         | 500-2000 km |
| 6m           | 50-54 MHz    | Sporadic-E      | 50-2000 km  |
| 2m (VHF)     | 144-148 MHz  | Line-of-sight   | 5-200 km    |
| 70cm (UHF)   | 420-450 MHz  | Line-of-sight   | 5-80 km     |

### HF Exercises
- Propagation: Solar activity, time factors.
- Projects: Monitor flux, contacts, antenna comparison, AI forecasting.

### VHF/UHF
- Horizon: d (km) = 4.12 × (√h₁ + √h₂).
- Exercises: Range testing, repeaters, digital modes.

### Antenna Design

| Type               | Formula (m) | Example (146 MHz) |
|--------------------|-------------|-------------------|
| Half-wave dipole   | 143 / f     | 0.979             |
| Quarter-wave       | 71.5 / f    | 0.490             |
| 5/8-wave           | 179 / f     | 1.226             |

---

## 5. Artificial Intelligence

### Propagation Prediction
- Inputs: Time, solar flux, frequency.
- Model: RandomForestClassifier for success probability.

```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100)
# Train and predict example
```

### Interference Classification

| Type             | Characteristics     | Mitigation          |
|------------------|---------------------|---------------------|
| Power Line       | 50/60 Hz buzz       | Directional antenna |
| TV/LED           | Broadband hash      | Source ID           |
| Wi-Fi            | Bursts              | Avoid frequency     |

### Message Routing
```python
class MessageRouter:
    def priority_score(self, message):
        return message.urgency * 0.5 + message.source_trust * 0.3 + (1 / message.age_minutes) * 0.2
```

### Antenna Optimization
Genetic algorithms: Evolve elements for gain, low SWR.

---

## 6. Advanced Coding: C++ & Python

### Python for RF
Spectrum Analyzer:
```python
from rtlsdr import RtlSdr
class SpectrumAnalyzer:
    def capture_spectrum(self):
        # FFT and power calc
```

Libraries: NumPy, SciPy, Matplotlib.

### C++ for DSP
FIR Filter:
```cpp
class FIRFilter {
    float process(float input) {
        // Convolution logic
    }
};
```

### Integration
SDR System: C++ processing, Python GUI.

---

## 7. Hands-On Projects

### Antennas
- VHF Dipole: 49.5 cm elements, SWR <2:1.
- Yagi: Reflector 104 cm, etc.; 10-12 dBi gain.

### RF Filter
- 2m Bandpass: Chebyshev with capacitors/inductors.

### SDR Setup
Hardware: RTL-SDR, Pi. Software: GNU Radio.

### Solar Station
Assembly: Panel mount, controller, battery wiring.

---

## 8. Field Operations

### Field Day Structure

| Time      | Activity                  |
|-----------|---------------------------|
| 08:00-09:30 | Setup                     |
| 09:30-10:00 | Testing                   |
| 10:00-12:00 | VHF/UHF                   |
| 12:00-13:00 | Lunch                     |
| 13:00-15:00 | HF                        |
| 15:00-16:00 | Analysis                  |
| 16:00-17:00 | Teardown                  |

### Challenges
- Distance: Power limits, scoring by km.
- Relay: Multi-hop timing.
- Solar-Only: Budget management.

### Annual Games
- Marathon: 24-hour rotations.
- Antenna Build: 2-hour competition.
- Fox Hunt: Transmitter location.

### Safety
- RF: Distance from antennas.
- Electrical: Protections, extinguishers.

---

## 9. Awards System

### Performance-Based
- **Tier 1**: Components (resistors, cables).
- **Tier 2**: Kits (transceivers, SDR).
- **Tier 3**: Stations (HF, solar kits).

### Sustainability
- Larger cohorts for resources.
- Community sponsorships.
- Equipment sharing.

---

## Conclusion

**RadioSport advances STEM education**—hands-on radio, energy, AI, development.

### Why It Matters
- Skills: Deploy systems.
- Thinking: Analytical.
- Confidence: Contacts.
- Community: Global.
- Careers: Technical.

### Join RadioSport
**Build. Test. Perfect.**

### Certificate
- 80% attendance.
- Projects complete.
- 6+ field days.
- Capstone.
- Safe ops.

📡 **RADIOSPORT**  
Building Tomorrow's Engineers  
Version 2.0

© RadioSport Program. Non-Commercial Educational Use.
