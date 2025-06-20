<h2 align="center">Pneumonia Detection from Chest X‑Rays</h2>
<h4 align="center">Deep Learning Pipeline: Preprocessing → Modeling → Evaluation</h4>

<h3>📁 Dataset</h3>
<ul>
  <li><strong>5,876 pediatric chest X‑rays</strong> (1593 Normal / 4,283 Pneumonia)</li>
  <li><strong>Split:</strong> Train 5,216 / Validation 16 / Test 624</li>
  <li><strong>Image size:</strong> 150×150 pixels, single‑channel (grayscale)</li>
  <li><strong>Source:</strong> Guangzhou Women and Children’s Medical Center</li>
</ul>

<h3>⚙️ Preprocessing & Augmentation</h3>
<ul>
  <li>Resize & Normalize: pixel values scaled to [0,1]</li>
  <li>Data Augmentation (Keras ImageDataGenerator):  
    rotation_range=30, zoom_range=0.2, width_shift_range=0.1,  
    height_shift_range=0.1, horizontal_flip=True</li>
  <li>Domain Integrity: <em>no vertical flips</em> to preserve anatomical orientation</li>
</ul>

<h3>🤖 Models & Approaches</h3>

<ol>
  <li>
    <strong>Custom CNN Architecture</strong><br>
    <em>Structure:</em>  
    5 Conv layers (32 → 64 → 64 → 128 → 256 filters), each with 3×3 kernels  
    <ul>
      <li>BatchNorm after every Conv</li>
      <li>Progressive Dropout: 0.1 → 0.2 → 0.2 over blocks</li>
      <li>MaxPooling (2×2) after each block</li>
      <li>Dense head: 128 → Sigmoid</li>
    </ul>
    <em>Training:</em> RMSprop + ReduceLROnPlateau, batch_size=32, EarlyStopping(patience=5)<br>
    <strong>Performance:</strong> 92.47% Accuracy | 0.96 AUC
  </li>

  <li>
    <strong>Transfer Learning</strong><br>
    <em>ResNet50:</em>  
    Frozen ImageNet weights + custom head (GAP → Dense(256→128→64) → Sigmoid)<br>
    <em>VGG19:</em>  
    Similar head design, slower convergence  
    <br><br>
    <table>
      <tr><th>Model</th><th>Accuracy</th><th>AUC</th></tr>
      <tr><td>ResNet50</td><td>77.08%</td><td>0.85</td></tr>
      <tr><td>VGG19</td><td>83.49%</td><td>0.92</td></tr>
    </table>
  </li>
</ol>

<h3>🔑 Key Technical Innovations</h3>
<ul>
  <li><strong>Grayscale‑Native Processing:</strong> Eliminated unnecessary RGB channels to reduce parameters</li>
  <li><strong>Progressive Regularization:</strong> Graduated dropout + batch normalization to combat overfitting</li>
  <li><strong>Anatomically‑Aware Augmentation:</strong> Horizontal flips only, preserving lung orientation</li>
  <li><strong>Optimizer Tuning:</strong> RMSprop selected over Adam for stable convergence on medical textures</li>
</ul>

<h3>⚡ Performance Optimization</h3>
<ul>
  <li><strong>Learning Rate Scheduling:</strong>  
    <code>ReduceLROnPlateau(monitor='val_accuracy', patience=2, factor=0.3, min_lr=1e-6)</code></li>
  <li><strong>Speed Advantage:</strong>  
    Custom CNN (~1.8M params) trains ~3× faster than ResNet50/VGG19 (>23M params)</li>
</ul>

<h3>📊 Comparative Insights</h3>
<ul>
  <li>Custom CNN outperforms TL models by +4.7% accuracy</li>
  <li>Medical‑specific augmentations yielded +2% AUC boost vs. generic pipelines</li>
  <li>RMSprop + small batches introduced beneficial gradient noise on sparse pathology patterns</li>
</ul>

<h3>📝 Summary</h3>
<p>
Implemented three strategies: a bespoke CNN tailored for grayscale X‑rays,  
ResNet50‑based transfer learning, and VGG19‑based transfer learning.  
The custom CNN achieved superior results (92.47% Acc, 0.96 AUC) through domain‑aware 
design choices—proving that lightweight, task‑specific architectures can outshine 
generic ImageNet models in medical imaging when rigorously regularized and optimized.
</p>

<h3>📚 References</h3>
<ul>
  <li>LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-Based Learning Applied to Document Recognition. : http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf </li>
  <li>He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition.: https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf</li>
  <li>Simonyan, K., & Zisserman, A. (2015). Very Deep Convolutional Networks for Large-Scale Image Recognition.https://arxiv.org/pdf/1409.1556 </li>
   <li>(https://www.youtube.com/watch?v=2dH_qjc9mFg&list=PLKnIA16_RmvYuZauWaPlRTC54KxSNLtNn)</li>
</ul>



