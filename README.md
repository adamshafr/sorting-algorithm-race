## Algorithm Name
Although my application allows the user to race and visualize several sorting algorithms (Bubble Sort, Selection Sort, Insertion Sort, Merge Sort, Quicksort, and Quicksort with Median of Three), the chosen algorithm I analyze for this project is Quicksort (Median of Three).
I originally built the app using only Quicksort, and later added more algorithms for comparison and visualization since the extension was straightforward, and I thought it would be cool to compare sorting algorithms side by side.
I chose Quicksort originally because it is my favorite sorting algorithm, I feel like it is pretty simple and super efficient. I also like the thought that has to go into pivot selection, and find it interesting how using one pivot selection strategy compares to using another.
For the purposes of this assignment, all problem breakdown, computational thinking, and algorithm explanation will focus only on the Median of Three Quicksort.

## Demo video/gif/screenshot of test

## Test Case 1: Quick Sort Vs Merge Sort (with small custom array) (Slow and detailed):
![GIF_1](https://github.com/user-attachments/assets/57d829df-19e8-438f-8f5d-54e28dbb0385)

## Test Case 2: Quick Sort Vs Merge sort (with big generated array) (faster):
![GIF_2](https://github.com/user-attachments/assets/2b59d711-2c86-405a-b25e-6b8f21b0d082)


## Test Case 3: Quick Sort Vs Selection sort (with many duplicates):
![GIF_3](https://github.com/user-attachments/assets/d8dbb7ee-8f6f-4703-8503-fcf7bd4f2930)



## Problem Breakdown & Computational Thinking 
The algorithm is: Quicksort (Median of Three)
The app also includes other sorting algorithms for comparison, but the following analysis focuses on the Median of Three Quicksort.

## Decomposition:
  To solve this problem, I broke it down into several smaller steps. First, the program needs to accept input from the user, either as a custom comma separated list or as a randomly generated list of integers. This input must be validated and converted into a usable Python list. Next, the Quicksort (Median of Three) algorithm is applied to the array, but in small stages: selecting the pivot, partitioning the current range, swapping elements when necessary, and then sorting the left and right subranges. At the same time, the visualization system captures key actions such as comparisons and swaps so the user can see how the algorithm behaves. Finally, after the sorting is complete, the app assembles all of the recorded frames into a video and displays it in the Gradio interface. Breaking the program into these steps made it easier to implement, test, and improve.

## Pattern Recognition:
  Quicksort works through clear patterns that repeat throughout the entire process. Each section of the array follows the exact same structure: choose a pivot, compare all elements to the pivot, swap elements when needed, and then split the problem into two smaller subproblems. This pattern continues until all sections reach a size of one. The Median-of-Three strategy also follows a repeating pattern of comparing the first, middle, and last elements to pick the pivot. From a visualization standpoint, the animation highlights these patterns clearly as comparisons appear as repeated color flashes, and swaps appear regularly whenever a smaller element needs to be moved to the left side. Recognizing these patterns helped guide both the code and the visual design.

## Abstraction:
  To keep the visualization easy to follow, I focused only on the key parts of the sorting process that matter most to understanding the algorithm. The animation shows comparisons, swaps, pivot selection, and the gradual progression toward a sorted array. Details such as index arithmetic, stack operations, and low-level Python behavior are intentionally hidden because they would clutter the visualization and distract from the main idea. When building the project, I originally focused on making just one sorting algorithm work correctly (the Median of Three Quicksort). I got that core algorithm working first, verified it, and then focused on the animation and visualization layer afterward. Once everything worked smoothly for one algorithm, it became very easy to add additional sorting algorithms later since the visual system was already designed.

## Algorithm Design:
  The overall flow of the program can be described as: input -> sorting -> visualization -> output. The user begins by entering a list or selecting a size for a random array. The program validates the input and then starts the Median-of-Three Quicksort process. For each subarray, it picks a pivot by taking the median of the first, middle, and last elements. That pivot is moved to the end, and the algorithm scans through the range, comparing each element to the pivot and swapping when needed. These actions are recorded as frames for the animation. Once the pivot is placed in its correct position, the left and right halves of the array are added to a stack to be sorted next, repeating the same steps until the entire array is sorted. After finishing, the program adds a short completion animation and then compiles all frames into a video, which is shown to the user through the Gradio interface. This structure keeps the logic simple and ensures the visualization matches the actual operations of the algorithm.

## Flowchart (For median of three + video generation):
<img width="672" height="888" alt="flowchart" src="https://github.com/user-attachments/assets/d8fa14af-cfc0-4914-8719-f7f2042e3d6e" />



## Time complexity of Algorithms:
    Insertion Sort - O(n2)
    Bubble Sort - O(n2)
    Selection Sort - O(n2)
    Merge Sort - O(n log n)
    Quick Sort - O(n log n)

## Steps to Run
1. Go to the Hugging Face link below.
2. Enter a custom array, or leave it empty and drag the slider to generate a random array of a certain size (100-1000)
3. Pick two sorting algorithms to race (it is recommended not to pick one of the O(n2) algorithms for a huge dataset, as it will take a while to process, but it will not take too long).
4. Click “Start Race”, wait for it to process, then watch the video of your two chosen algorithms sorting the same array.
5. You can watch which one is faster, and visualize the differences between how these sorting algorithms work.

## Hugging Face Link 
https://huggingface.co/spaces/AdamShafr/sorting-race

## Author & Acknowledgment
This project was created by Adam Shafronsky for the Sorting Visualization assignment.
I wrote all core code for the sorting algorithms and implemented the overall logic of the visualizer.
AI tools were used to assist with parts of the UI and frame-generation structure, but all tuning, optimization, debugging, and final refinements to ensure the program works efficiently and appears visually polished were done by me.

