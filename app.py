import gradio as gr
import random
import numpy as np
import cv2
import tempfile
import os
import atexit
from typing import List, Tuple, Callable, Optional
import math

FPS = 30  # frames per second for video
_temp_files = []  # list to store temporary video files


# -------------------------------------------------------------
#                   FRAME RENDERING
# -------------------------------------------------------------
def render_frame(array: List[int], highlights: Tuple[int, ...] = (), action: str = "") -> np.ndarray:
    #Render a single frame of the array as bars with optional highlights
    h, w = 300, len(array)  # height and width of image
    img = np.zeros((h, w, 3), dtype=np.uint8)  # empty black image
    
    if not array:
        return img  # return empty image if array is empty
    
    max_val = max(array)
    bar_heights = ((np.array(array) / max_val) * (h - 1)).astype(int)  # scale bars to height
    
    # Draw normal light blue bars
    for x, height in enumerate(bar_heights):
        img[h - height:h, x] = (180, 200, 255)
    
    # Highlight specific bars for compare, swap, or place actions
    if highlights:
        if action == "compare":
            color = (255, 255, 0)  # yellow
        elif action == "swap":
            color = (0, 255, 0)  # green
        elif action == "place":
            color = (255, 0, 0)  # red
        else:  # finished
            color = (255, 128, 0)  # orange/blue
        for x in highlights:
            if 0 <= x < w:
                height = bar_heights[x]
                img[h - height:h, x] = color
    
    return img


def create_finish_animation(array: List[int], num_frames: int = 10) -> List[np.ndarray]:
    #Create a simple finishing animation with dark blue bars
    if not array:
        return []

    frames = []
    h, w = 300, len(array)
    max_val = max(array)
    
    for frame_num in range(num_frames):
        progress = (frame_num + 1) / num_frames
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        for x, val in enumerate(array):
            height = int((val / max_val) * (h - 1))
            # Dark blue for completed part
            if (x + 1) / w <= progress:
                img[h - height:h, x] = (80, 100, 200)
            else:  # light blue for remaining
                img[h - height:h, x] = (180, 200, 255)
        
        frames.append(img)
    
    return frames


# -------------------------------------------------------------
#                   SORTING VISUALIZER CLASS
# -------------------------------------------------------------
class SortingVisualizer:
    def __init__(self, n: int, algo_name: str):
        self.n = n
        self.algo_name = algo_name
        self.steps = 0  # step counter for skipping frames
        self.frames = []  # store captured frames
        self.skip = self.getFastSkip()  # determine how often to capture frames
    
    def getFastSkip(self) -> int:
        #Decide frame skipping based on array size
        #Skips are important because it prevents large arrays video from being too long, and the processing being too long. This way it should never take over 20 seconds
        #to process any given video
        n = self.n
        if n < 100:
            return 1 #For small arrays every detail is shown slowly.
        elif n <= 300:
            return 50
        elif n <= 600:
            return 75
        else:
            return 80
    
    def should_capture(self) -> bool:
        #Determine whether this step should be captured
        self.steps += 1
        return self.steps % self.skip == 0
    
    def capture_frame(self, arr: List[int], highlights: Tuple[int, ...] = (), action: str = ""):
        #Capture a frame if needed
        if self.should_capture():
            self.frames.append(render_frame(arr, highlights, action))


# -------------------------------------------------------------
#                   SORTING ALGORITHMS
# -------------------------------------------------------------
def bubble_sort(arr: List[int], viz: SortingVisualizer) -> List[int]:
    #Simple bubble sort with visualization
    n = len(arr)
    for i in range(n):
        swapped = False
        for j in range(n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]  # swap values
                swapped = True
                viz.capture_frame(arr, (j, j + 1), "swap")  # show swap
            else:
                viz.capture_frame(arr, (j, j + 1), "compare")  # show comparison
        if not swapped:
            break  # already sorted
    return arr


def selection_sort(arr: List[int], viz: SortingVisualizer) -> List[int]:
    #Selection sort with visualization
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
            if j % 3 == 0:
                viz.capture_frame(arr, (min_idx, j), "compare")
        if min_idx != i:
            arr[i], arr[min_idx] = arr[min_idx], arr[i]  # swap minimum
            viz.capture_frame(arr, (i, min_idx), "swap")
    return arr


def insertion_sort(arr: List[int], viz: SortingVisualizer) -> List[int]:
    #Insertion sort with visualization
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        moved = False
        while j >= 0 and key < arr[j]:
            arr[j + 1] = arr[j]  # shift element right
            moved = True
            viz.capture_frame(arr, (j, j + 1), "compare")
            j -= 1
        arr[j + 1] = key  # insert key
        if moved:
            viz.capture_frame(arr, (j + 1,), "place")
    return arr


def quicksort(arr: List[int], viz: SortingVisualizer) -> List[int]:
    #Iterative quicksort with visualization
    stack = [(0, len(arr) - 1)]
    while stack:
        low, high = stack.pop()
        if low < high:
            pivot = arr[high]
            i = low - 1
            for j in range(low, high):
                viz.capture_frame(arr, (j, high), "compare")
                if arr[j] <= pivot:
                    i += 1
                    if i != j:
                        arr[i], arr[j] = arr[j], arr[i]
                        viz.capture_frame(arr, (i, j), "swap")
            arr[i + 1], arr[high] = arr[high], arr[i + 1]
            viz.capture_frame(arr, (i + 1, high), "swap")
            stack.append((low, i))
            stack.append((i + 2, high))
    return arr


def quicksort_median3(arr: List[int], viz: SortingVisualizer) -> List[int]:
    #Quicksort with median of three pivot
    def median_of_three(lo, hi):
        mid = (lo + hi) // 2
        a, b, c = arr[lo], arr[mid], arr[hi]
        viz.capture_frame(arr, (lo, mid), "compare")
        viz.capture_frame(arr, (mid, hi), "compare")
        # return median index
        if a <= b <= c or c <= b <= a:
            return mid
        elif b <= a <= c or c <= a <= b:
            return lo
        else:
            return hi

    stack = [(0, len(arr) - 1)]
    while stack:
        low, high = stack.pop()
        if low < high:
            pivot_index = median_of_three(low, high)
            arr[pivot_index], arr[high] = arr[high], arr[pivot_index]
            viz.capture_frame(arr, (pivot_index, high), "swap")
            pivot = arr[high]
            i = low - 1
            for j in range(low, high):
                viz.capture_frame(arr, (j, high), "compare")
                if arr[j] <= pivot:
                    i += 1
                    if i != j:
                        arr[i], arr[j] = arr[j], arr[i]
                        viz.capture_frame(arr, (i, j), "swap")
            arr[i + 1], arr[high] = arr[high], arr[i + 1]
            viz.capture_frame(arr, (i + 1, high), "swap")
            stack.append((low, i))
            stack.append((i + 2, high))
    return arr


def merge_sort(arr: List[int], viz: SortingVisualizer) -> List[int]:
    #Merge sort with visualization
    def _merge(l: int, mid: int, r: int):
        left, right = arr[l:mid + 1], arr[mid + 1:r + 1]
        i = j = 0
        k = l
        while i < len(left) and j < len(right):
            viz.capture_frame(arr, (l + i, mid + 1 + j), "compare")
            if left[i] <= right[j]:
                arr[k] = left[i]
                i += 1
            else:
                arr[k] = right[j]
                j += 1
            viz.capture_frame(arr, (k,), "place")
            k += 1
        while i < len(left):
            arr[k] = left[i]
            viz.capture_frame(arr, (k,), "place")
            i += 1
            k += 1
        while j < len(right):
            arr[k] = right[j]
            viz.capture_frame(arr, (k,), "place")
            j += 1
            k += 1

    def _sort(l: int, r: int):
        if l < r:
            mid = (l + r) // 2
            _sort(l, mid)
            _sort(mid + 1, r)
            _merge(l, mid, r)
    
    _sort(0, len(arr) - 1)
    return arr


# -------------------------------------------------------------
#                 RACE SYSTEM
# -------------------------------------------------------------
def run_algorithm(name: str, arr: List[int]) -> List[np.ndarray]:
    #Run a sorting algorithm and capture frames
    viz = SortingVisualizer(len(arr), name)
    algorithms = {
        "Bubble Sort": bubble_sort,
        "Selection Sort": selection_sort,
        "Insertion Sort": insertion_sort,
        "Quicksort": quicksort,
        "Quicksort (Median of Three)": quicksort_median3,
        "Merge Sort": merge_sort
    }
    arr_copy = arr.copy()
    algorithms[name](arr_copy, viz)

    # Ensure at least one frame exists
    if len(viz.frames) == 0:
        viz.frames.append(render_frame(arr_copy))

    # Add finish animation
    viz.frames.extend(create_finish_animation(arr_copy))
    return viz.frames


def generate_race_video(algo1: str, algo2: str, custom_array: str, n: int) -> str:
    #Generate video showing two sorting algorithms racing
    # Parse custom array or generate random
    if custom_array.strip():
        try:
            base_arr = [int(x.strip()) for x in custom_array.split(",") if x.strip()]
            if len(base_arr) == 0:
                raise ValueError
        except:
            return "ERROR: Invalid custom array. Please enter comma-separated integers."
    else:
        base_arr = list(range(1, n + 1))
        random.shuffle(base_arr)

    frames1 = run_algorithm(algo1, base_arr)
    frames2 = run_algorithm(algo2, base_arr)

    # Ensure both have at least 1 frame
    if not frames1:
        frames1 = [render_frame(base_arr)]
    if not frames2:
        frames2 = [render_frame(base_arr)]

    max_frames = max(len(frames1), len(frames2))

    # Pad shorter one
    if len(frames1) < max_frames:
        frames1.extend([frames1[-1]] * (max_frames - len(frames1)))
    if len(frames2) < max_frames:
        frames2.extend([frames2[-1]] * (max_frames - len(frames2)))

    # Combine frames horizontally with separator
    h, w1, _ = frames1[0].shape
    w2 = frames2[0].shape[1]
    space_width = 12
    separator_width = 2
    space = np.zeros((h, space_width, 3), dtype=np.uint8)
    separator = np.full((h, separator_width, 3), (255, 0, 0), dtype=np.uint8)

    race_frames = []
    for f1, f2 in zip(frames1, frames2):
        if f1.shape != (h, w1, 3):
            f1 = cv2.resize(f1, (w1, h))
        if f2.shape != (h, w2, 3):
            f2 = cv2.resize(f2, (w2, h))
        race_frames.append(np.hstack([f1, space, separator, space, f2]))

    return create_video(race_frames)


# -------------------------------------------------------------
#                 VIDEO HANDLING + CLEANUP
# -------------------------------------------------------------
def cleanup_temp_files():
    #Delete temporary video files
    global _temp_files
    for temp_file in _temp_files:
        try:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
        except:
            pass
    _temp_files.clear()


def create_video(frames: List[np.ndarray]) -> str:
    #Create video from frames, try multiple codecs
    if not frames:
        return ""

    # Ensure all frames same size
    base_h, base_w = frames[0].shape[:2]
    safe_frames = []
    for f in frames:
        if f.shape[:2] != (base_h, base_w):
            f = cv2.resize(f, (base_w, base_h))
        safe_frames.append(f)
    frames = safe_frames
    h, w, _ = frames[0].shape

    for codec in ['mp4v', 'avc1', 'X264', 'MJPG']:
        try:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            temp_path = temp_file.name
            temp_file.close()
            global _temp_files
            _temp_files.append(temp_path)
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(temp_path, fourcc, FPS, (w, h))
            if out.isOpened():
                for frame in frames:
                    out.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                out.release()
                print(f"Successfully created video with codec: {codec}")
                return temp_path
        except Exception as e:
            print(f"Codec {codec} failed: {e}")
            continue
    
    # Final fallback AVI
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.avi')
        temp_path = temp_file.name
        temp_file.close()
        _temp_files.append(temp_path)
        out = cv2.VideoWriter(temp_path, cv2.VideoWriter_fourcc(*'DIVX'), FPS, (w, h))
        for frame in frames:
            out.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        out.release()
        return temp_path
    except Exception as e:
        print(f"Final fallback failed: {e}")
        return ""


atexit.register(cleanup_temp_files)  # cleanup files on exit


# -------------------------------------------------------------
#                 GRADIO UI
# -------------------------------------------------------------
with gr.Blocks(title="Sorting Algorithm Race") as demo:
    gr.Markdown("""
    # 🏁 Sorting Algorithm Race
    **Pick two sorting algorithms and see which one is faster!**
    - Enter a custom array (comma-separated), *or*
    - Leave it blank and use the slider to generate a random array
    """)

    algo_list = [
        "Bubble Sort",
        "Selection Sort",
        "Insertion Sort",
        "Quicksort",
        "Quicksort (Median of Three)",
        "Merge Sort"
    ]

    with gr.Row():
        algo1 = gr.Dropdown(algo_list, value="Quicksort", label="Algorithm 1")
        algo2 = gr.Dropdown(algo_list, value="Quicksort (Median of Three)", label="Algorithm 2")

    custom_array = gr.Textbox(label="Custom Array (optional)",
                              placeholder="Example: 5, 1, 4, 2, 3",
                              lines=1)

    n_race = gr.Slider(50, 1000, 300, step=10,
                       label="Size of RANDOM array (only used if no custom array is provided)")

    race_btn = gr.Button("🎬 Start Race", variant="primary", size="lg")
    
    race_video = gr.Video(autoplay=True, label="Race Result", height=400)

    race_btn.click(fn=generate_race_video,
                   inputs=[algo1, algo2, custom_array, n_race],
                   outputs=race_video)


if __name__ == "__main__":
    try:
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        print("Starting Sorting Algorithm Race...")
        demo.launch(show_error=True)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cleanup_temp_files()
