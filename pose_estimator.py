#!/usr/bin/env python3
"""
Pose Estimator using MediaPipe and OpenCV
Detects human pose landmarks and draws skeleton overlay on video.
"""

import cv2
import mediapipe as mp
import argparse
from pathlib import Path


class PoseEstimator:
    """Pose estimation using MediaPipe Pose solution."""

    # Face landmark indices (0-10): nose, eyes, ears, mouth
    FACE_LANDMARKS = set(range(11))

    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5,
                 exclude_face=True):
        """
        Initialize the pose estimator.

        Args:
            min_detection_confidence: Minimum confidence for pose detection (0.0-1.0)
            min_tracking_confidence: Minimum confidence for pose tracking (0.0-1.0)
            exclude_face: If True, exclude face landmarks (eyes, ears, nose, mouth)
        """
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.exclude_face = exclude_face

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        # Custom drawing specs for better visibility
        self.landmark_style = self.mp_drawing.DrawingSpec(
            color=(0, 255, 0),  # Green joints
            thickness=2,
            circle_radius=3
        )
        self.connection_style = self.mp_drawing.DrawingSpec(
            color=(255, 0, 0),  # Blue bones (BGR format)
            thickness=2
        )

        # Filter connections to exclude face if requested
        if exclude_face:
            self.connections = frozenset(
                (a, b) for a, b in self.mp_pose.POSE_CONNECTIONS
                if a not in self.FACE_LANDMARKS and b not in self.FACE_LANDMARKS
            )
        else:
            self.connections = self.mp_pose.POSE_CONNECTIONS

    def process_frame(self, frame):
        """
        Process a single frame and detect pose landmarks.

        Args:
            frame: BGR image from OpenCV

        Returns:
            Annotated frame with pose skeleton drawn
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame
        results = self.pose.process(rgb_frame)

        # Draw pose landmarks if detected
        if results.pose_landmarks:
            if self.exclude_face:
                # Custom drawing to exclude face landmarks
                self._draw_landmarks_excluding_face(frame, results.pose_landmarks)
            else:
                self.mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.landmark_style,
                    connection_drawing_spec=self.connection_style
                )

        return frame, results

    def _draw_landmarks_excluding_face(self, frame, landmarks):
        """Draw landmarks excluding face points."""
        h, w = frame.shape[:2]

        # Draw connections (bones)
        for connection in self.connections:
            start_idx, end_idx = connection
            start = landmarks.landmark[start_idx]
            end = landmarks.landmark[end_idx]

            start_point = (int(start.x * w), int(start.y * h))
            end_point = (int(end.x * w), int(end.y * h))

            cv2.line(frame, start_point, end_point,
                     self.connection_style.color, self.connection_style.thickness)

        # Draw landmarks (joints) - skip face landmarks
        for idx, landmark in enumerate(landmarks.landmark):
            if idx in self.FACE_LANDMARKS:
                continue
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(frame, (x, y), self.landmark_style.circle_radius,
                       self.landmark_style.color, self.landmark_style.thickness)

    def process_video(self, input_path, output_path=None, show_preview=False):
        """
        Process an entire video file.

        Args:
            input_path: Path to input video file
            output_path: Path to output video file (optional)
            show_preview: Whether to show live preview window

        Returns:
            True if processing completed successfully
        """
        input_path = Path(input_path)
        if not input_path.exists():
            print(f"Error: Input file not found: {input_path}")
            return False

        # Open video capture
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            print(f"Error: Could not open video: {input_path}")
            return False

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Input video: {input_path.name}")
        print(f"Resolution: {width}x{height}, FPS: {fps:.2f}, Frames: {total_frames}")

        # Setup video writer if output path specified
        writer = None
        if output_path:
            output_path = Path(output_path)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            print(f"Output video: {output_path}")

        frame_count = 0
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Process frame
                annotated_frame, _ = self.process_frame(frame)

                # Write to output
                if writer:
                    writer.write(annotated_frame)

                # Show preview
                if show_preview:
                    cv2.imshow('Pose Estimation', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\nPreview closed by user")
                        break

                frame_count += 1
                if frame_count % 100 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")

        finally:
            cap.release()
            if writer:
                writer.release()
            if show_preview:
                cv2.destroyAllWindows()

        print(f"Processed {frame_count} frames")
        if output_path:
            print(f"Output saved to: {output_path}")

        return True

    def close(self):
        """Release resources."""
        self.pose.close()


def main():
    parser = argparse.ArgumentParser(description='Pose Estimation using MediaPipe')
    parser.add_argument('input', help='Input video file path')
    parser.add_argument('-o', '--output', help='Output video file path')
    parser.add_argument('--preview', action='store_true', help='Show live preview')
    parser.add_argument('--detection-confidence', type=float, default=0.5,
                        help='Minimum detection confidence (0.0-1.0)')
    parser.add_argument('--tracking-confidence', type=float, default=0.5,
                        help='Minimum tracking confidence (0.0-1.0)')
    parser.add_argument('--with-face', action='store_true',
                        help='Include face landmarks (eyes, ears, nose, mouth)')

    args = parser.parse_args()

    # Create pose estimator
    estimator = PoseEstimator(
        min_detection_confidence=args.detection_confidence,
        min_tracking_confidence=args.tracking_confidence,
        exclude_face=not args.with_face
    )

    try:
        success = estimator.process_video(
            args.input,
            args.output,
            show_preview=args.preview
        )
        return 0 if success else 1
    finally:
        estimator.close()


if __name__ == '__main__':
    exit(main())
