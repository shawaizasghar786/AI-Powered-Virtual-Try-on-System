# Input Image (256×192×3)
# ↓
# Encoder Level 1 (128×96×64)
# ↓
# Encoder Level 2 (64×48×128)
# ↓
# Encoder Level 3 (32×24×256)
# ↓
# Encoder Level 4 (16×12×512)
# ↓
# Middle Bridge (8×6×1024)
# ↓
# Decoder Level 4 (16×12×512)
# ↓
# Decoder Level 3 (32×24×256)
# ↓
# Decoder Level 2 (64×48×128)
# ↓
# Decoder Level 1 (128×96×64)
# ↓
# Final Output (256×192×1)


# graph TD
#     A[Capture Frame] --> B[Detect Pose]
#     B --> C[Calculate Shirt Position]
#     C --> D[Resize & Position Shirt]
#     D --> E[Blend Shirt with Frame]
#     E --> F[Update Display]
#     F --> A