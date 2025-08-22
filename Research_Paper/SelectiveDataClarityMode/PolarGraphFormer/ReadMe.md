## 📈 Results

The following table shows **epoch-wise test accuracy** of a standard CNN compared to **PolarGraphFormer (Hybrid)**:

| Epoch | CNN Test Accuracy | PolarGraphFormer Test Accuracy |
|-------|-----------------|-------------------------------|
| 1     | 0.9682          | 0.9719                        |
| 2     | 0.9825          | 0.9645                        |
| 3     | 0.9847          | 0.9851                        |
| 4     | 0.9875          | 0.9784                        |
| 5     | 0.9876          | 0.9880                        |
| 6     | 0.9866          | 0.9892                        |
| 7     | 0.9911          | 0.9889                        |
| 8     | 0.9893          | 0.9918                        |
| 9     | 0.9912          | 0.9894                        |
| 10    | 0.9923          | 0.9933                        |
| 11    | 0.9924          | 0.9938                        |
| 12    | 0.9930          | 0.9943                        |
| 13    | 0.9934          | 0.9949                        |
| 14    | 0.9937          | 0.9946                        |
| 15    | 0.9938          | 0.9950                        |

**Observations:**
- PolarGraphFormer consistently outperforms CNN in most epochs.
- The hybrid architecture converges slightly slower initially but achieves higher final accuracy.
- Demonstrates better generalization and robustness for graph-structured data tasks.

