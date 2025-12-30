# Sorting Algorithm Race

An interactive Python application that visualizes and compares common sorting algorithms in real time, allowing users to see how algorithmic complexity translates into practical performance.

## Live Demo
**Deployed demo:** https://huggingface.co/spaces/AdamShafr/sorting-race

---

## Overview
Sorting Algorithm Race is a visual and interactive tool designed to help users understand how different sorting algorithms behave on the same dataset. Instead of focusing only on theoretical time complexity, this project highlights how performance differences appear during execution.

The application renders multiple algorithms side by side, updating each step in real time so users can directly compare speed, efficiency, and behaviour.

---

## Key Features
- Side by side visualisation of multiple sorting algorithms  
- Real time animation of sorting steps  
- Adjustable input size and test cases  
- Clear comparison of algorithm efficiency and behaviour  
- Deployed web demo accessible without local setup  

---

## Algorithms Implemented
- Bubble Sort  
- Selection Sort  
- Insertion Sort  
- Merge Sort  
- Quick Sort  

Each algorithm operates on identical input data to ensure a fair comparison.

---

## How It Works (High Level)
1. Input data is generated or selected by the user  
2. Each sorting algorithm runs independently on a copy of the same dataset  
3. Sorting steps are captured and rendered frame by frame  
4. Visual output updates in real time, allowing direct comparison  

The architecture separates algorithm logic from rendering logic to keep the code modular and readable.

---

## Run Locally

### Requirements
- Python 3.9 or newer  
- pip  

### Setup
```bash
git clone https://github.com/adamshafr/sorting-algorithm-race.git
cd sorting-algorithm-race
pip install -r requirements.txt
python main.py
```

## What I Learned Building This

- How theoretical algorithm complexity translates into real-world visual performance
- Tradeoffs between algorithm clarity and rendering efficiency
- Managing frame updates and timing across multiple concurrent processes
- Packaging and deploying a Python application for public use
- Debugging UI-driven logic and performance bottlenecks

---

## Future Improvements

- Add more advanced algorithms for comparison
- Improve performance metrics and timing overlays
- Add automated unit tests for algorithm validation
- Expand visual controls and configuration options

---

## Technologies Used

- Python
- Matplotlib
- NumPy
- Git and GitHub
- Hugging Face Spaces (deployment)

---

## Notes

This project was developed as part of a university computer science course and expanded beyond the original requirements to include deployment and improved usability.

AI tools were used to assist with parts of the UI and frame generation structure, but all tuning, optimization, debugging, and final refinements to ensure the program works efficiently and appears visually polished were done by the author.

