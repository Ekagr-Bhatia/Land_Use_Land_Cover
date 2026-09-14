/**
 * TERRA / VISION — Frontend Client Application
 * Communicates with the FastAPI inference engine for 13-band Sentinel-2 LULC classification.
 */

(function () {
  'use strict';

  // DOM Selectors
  const dropZone = document.getElementById('drop-zone');
  const fileInput = document.getElementById('file-input');
  const modelSelect = document.getElementById('model-select');
  const runBtn = document.getElementById('run-btn');
  const resetBtn = document.getElementById('reset-btn');

  // Inspector Elements
  const previewBox = document.getElementById('upload-preview-box');
  const detailFilename = document.getElementById('detail-filename');
  const detailSize = document.getElementById('detail-size');
  const detailBands = document.getElementById('detail-bands');
  const detailStatus = document.getElementById('detail-status');

  // Output Container
  const resultContainer = document.getElementById('result-container');

  // Dynamic Info Containers
  const bandsGrid = document.getElementById('bands-grid');
  const samplesContainer = document.getElementById('samples-container');
  const factFramework = document.getElementById('fact-framework');
  const factDevice = document.getElementById('fact-device');

  // Client State
  let currentFile = null;
  let currentSampleId = null;

  /**
   * Helper: Formats bytes to human-readable size
   */
  function formatBytes(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }

  /**
   * Helper: Formats class name (e.g., HerbaceousVegetation -> Herbaceous Vegetation)
   */
  function formatClassName(name) {
    if (!name) return '';
    return name
      .replace(/([a-z])([A-Z])/g, '$1 $2')
      .replace('Sea Lake', 'Sea & Lake')
      .toUpperCase();
  }

  /**
   * API Fetch wrapper with JSON error handling
   */
  async function apiRequest(endpoint, options = {}) {
    const response = await fetch(endpoint, options);
    if (!response.ok) {
      let errorMessage = `HTTP Error ${response.status}`;
      try {
        const errorData = await response.json();
        if (errorData.detail) errorMessage = errorData.detail;
      } catch (e) {
        // Fallback to text
      }
      throw new Error(errorMessage);
    }
    return response.json();
  }

  /**
   * Updates the Scene Telemetry Inspector with selected file
   */
  function updateInspector(file, previewUrl = null) {
    currentFile = file;
    currentSampleId = null;

    detailFilename.textContent = file.name;
    detailSize.textContent = formatBytes(file.size);
    detailBands.textContent = '13 Bands (Sentinel-2)';
    detailStatus.textContent = 'Ready for inference';
    detailStatus.className = 'detail-val status-ready';

    if (previewUrl) {
      previewBox.innerHTML = `<img src="${previewUrl}" alt="${file.name}">`;
    } else {
      previewBox.innerHTML = `
        <div class="empty-preview-state">
          <span class="crosshair">✓</span>
          <p>${file.name}</p>
          <small>${formatBytes(file.size)} · 13-band GeoTIFF</small>
        </div>
      `;
    }

    runBtn.disabled = false;
  }

  /**
   * Clears inspection and resets state
   */
  function resetState() {
    currentFile = null;
    currentSampleId = null;
    fileInput.value = '';

    detailFilename.textContent = '—';
    detailSize.textContent = '—';
    detailBands.textContent = '13 Bands (Expected)';
    detailStatus.textContent = 'Awaiting input';
    detailStatus.className = 'detail-val status-idle';

    previewBox.innerHTML = `
      <div class="empty-preview-state">
        <span class="crosshair">+</span>
        <p>NO SCENE LOADED</p>
        <small>Drag a .tif or select a sample</small>
      </div>
    `;

    runBtn.disabled = true;

    resultContainer.innerHTML = `
      <div class="pipeline-empty-state" id="pipeline-state">
        <div class="pipeline-steps">
          <span class="p-step">IMAGE</span>
          <span class="p-arr">↓</span>
          <span class="p-step">ANALYSIS</span>
          <span class="p-arr">↓</span>
          <span class="p-step">PATTERN</span>
          <span class="p-arr">↓</span>
          <span class="p-step">CLASSIFICATION</span>
        </div>
        <p class="pipeline-hint">Submit a 13-band Sentinel-2 GeoTIFF above or select a sample scene below.</p>
      </div>
    `;
  }

  /**
   * Renders the loading animation in the Classification section
   */
  function renderLoadingState() {
    resultContainer.innerHTML = `
      <div class="pipeline-loading-state">
        <div class="loading-steps">
          <span class="loading-step active pulse">IMAGE INGESTION</span>
          <span class="loading-step">SPECTRAL NORMALIZATION (13 CHANNELS)</span>
          <span class="loading-step">TENSOR CONVOLUTION</span>
          <span class="loading-step">SOFTMAX CLASSIFICATION</span>
        </div>
        <p class="pipeline-hint" style="margin-top: 24px;">Executing neural forward pass…</p>
      </div>
    `;
    window.location.hash = 'classification';
  }

  /**
   * Renders an error message in the Classification section
   */
  function renderErrorState(errorMessage) {
    resultContainer.innerHTML = `
      <div class="pipeline-empty-state">
        <p class="section-tag" style="color: #e53935;">INFERENCE ERROR</p>
        <h3 style="font-size: 18px; color: var(--white); margin-bottom: 12px;">Failed to process GeoTIFF</h3>
        <p class="pipeline-hint" style="max-width: 500px; margin: 0 auto; color: #ff8a80;">${errorMessage}</p>
      </div>
    `;
    window.location.hash = 'classification';
  }

  /**
   * Renders the real classification result and probability distribution
   */
  function renderResult(result) {
    const formattedClass = formatClassName(result.predicted_class);
    const topProb = result.probabilities[0]?.label;

    // Update inspector preview thumbnail if preview data URI returned
    if (result.preview) {
      previewBox.innerHTML = `<img src="${result.preview}" alt="${result.predicted_class}">`;
    }

    const distributionHtml = result.probabilities
      .map((item) => {
        const isHighlight = item.label === topProb;
        const displayName = formatClassName(item.label);
        return `
          <div class="prob-row ${isHighlight ? 'highlight' : ''}">
            <span class="prob-label" title="${displayName}">${displayName}</span>
            <div class="prob-bar-track">
              <div class="prob-bar-fill" style="width: ${item.percentage}%"></div>
            </div>
            <span class="prob-val">${item.percentage.toFixed(2)}%</span>
          </div>
        `;
      })
      .join('');

    const previewSection = result.preview
      ? `
        <div class="prediction-preview-container">
          <img src="${result.preview}" alt="${formattedClass}" class="prediction-preview-img">
          <div class="preview-meta">
            <b>TRUE-COLOR COMPOSITE</b>
            <span>Sentinel-2 Bands 4, 3, 2 (RGB)</span>
            <span>Spatial Extent: 64 × 64 pixels</span>
          </div>
        </div>
      `
      : '';

    resultContainer.innerHTML = `
      <div class="classification-grid">
        <div class="prediction-primary-card">
          <span class="prediction-badge-tag">PREDICTED LAND COVER</span>
          <div class="prediction-class-name">${formattedClass}</div>
          <div class="prediction-confidence-pill">
            ${result.confidence.toFixed(2)}%
            <small>CONFIDENCE</small>
          </div>
          ${previewSection}
          <div class="telemetry-row">
            <span>MODEL: <b>${result.model_name || result.model}</b></span>
            <span>LATENCY: <b>${result.metadata?.latency_ms ?? '—'} ms</b></span>
            <span>CHANNELS: <b>13 MSI</b></span>
            <span>DEVICE: <b>${result.metadata?.device ?? 'CPU'}</b></span>
          </div>
        </div>

        <div class="distribution-container">
          <div class="dist-header">COMPLETE PROBABILITY DISTRIBUTION (10 CLASSES)</div>
          ${distributionHtml}
        </div>
      </div>
    `;

    window.location.hash = 'classification';
  }

  /**
   * Executes inference on the currently loaded file or sample
   */
  async function executeClassification() {
    if (!currentFile && !currentSampleId) return;

    const selectedModel = modelSelect.value;
    runBtn.disabled = true;
    runBtn.textContent = 'ANALYSING…';

    renderLoadingState();

    try {
      let result;
      if (currentSampleId) {
        // Run inference on sample scene
        result = await apiRequest(
          `/samples/${currentSampleId}/predict?model=${encodeURIComponent(selectedModel)}`,
          { method: 'POST' }
        );
      } else {
        // Upload and run inference on uploaded GeoTIFF
        const formData = new FormData();
        formData.append('file', currentFile);
        result = await apiRequest(
          `/predict?model=${encodeURIComponent(selectedModel)}`,
          {
            method: 'POST',
            body: formData
          }
        );
      }

      renderResult(result);
    } catch (err) {
      renderErrorState(err.message || 'An error occurred during inference.');
    } finally {
      runBtn.disabled = false;
      runBtn.textContent = 'RUN CLASSIFICATION →';
    }
  }

  /**
   * File Drag & Drop Listeners
   */
  function initDragAndDrop() {
    ['dragenter', 'dragover'].forEach((eventName) => {
      dropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.add('dragover');
      });
    });

    ['dragleave', 'drop'].forEach((eventName) => {
      dropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.remove('dragover');
      });
    });

    dropZone.addEventListener('drop', (e) => {
      const files = e.dataTransfer.files;
      if (files && files.length > 0) {
        const file = files[0];
        if (file.name.toLowerCase().endsWith('.tif') || file.name.toLowerCase().endsWith('.tiff')) {
          updateInspector(file);
        } else {
          alert('Please upload a 13-band Sentinel-2 GeoTIFF file (.tif or .tiff).');
        }
      }
    });

    fileInput.addEventListener('change', () => {
      if (fileInput.files && fileInput.files.length > 0) {
        updateInspector(fileInput.files[0]);
      }
    });

    // Make dropZone keyboard-accessible
    dropZone.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        fileInput.click();
      }
    });
  }

  /**
   * Initial Setup: Loads model info, Sentinel-2 bands, and sample library
   */
  async function initializeApp() {
    // 1. Fetch Model Info
    try {
      const info = await apiRequest('/model-info');

      if (factFramework) factFramework.textContent = info.framework;
      if (factDevice) factDevice.textContent = info.device;

      // Populate Model Select Options
      if (info.models && Object.keys(info.models).length > 0) {
        modelSelect.innerHTML = Object.entries(info.models)
          .map(([key, spec]) => {
            const isDefault = key === info.default_model;
            return `<option value="${key}" ${isDefault ? 'selected' : ''}>${spec.name} · ${spec.training} (${spec.params})</option>`;
          })
          .join('');
      }

      // Populate 13 Sentinel-2 Bands
      if (info.bands && bandsGrid) {
        bandsGrid.innerHTML = info.bands
          .map((b) => `
            <div class="band-pill">
              <span class="band-code">${b.name}</span>
              <span class="band-desc">${b.desc}</span>
            </div>
          `)
          .join('');
      }
    } catch (err) {
      console.warn('Failed to fetch /model-info:', err);
    }

    // 2. Fetch Sample Scenes
    try {
      const data = await apiRequest('/samples');
      if (data.samples && data.samples.length > 0) {
        samplesContainer.innerHTML = data.samples
          .map((sample) => `
            <button class="sample-card" data-id="${sample.id}" data-label="${sample.label}" data-filename="${sample.filename}" data-preview="${sample.preview}">
              <div class="sample-img-wrap">
                <img src="${sample.preview}" alt="${sample.label}" loading="lazy">
              </div>
              <strong class="sample-label">${formatClassName(sample.label)}</strong>
              <span class="sample-tag">13-BAND · GEOTIFF</span>
            </button>
          `)
          .join('');

        // Attach Click Handlers to Sample Cards
        document.querySelectorAll('.sample-card').forEach((card) => {
          card.addEventListener('click', async () => {
            const sampleId = card.dataset.id;
            const label = card.dataset.label;
            const filename = card.dataset.filename;
            const previewUrl = card.dataset.preview;

            currentSampleId = sampleId;
            currentFile = null;

            detailFilename.textContent = filename;
            detailSize.textContent = 'EuroSAT Scene';
            detailBands.textContent = '13 Bands MSI';
            detailStatus.textContent = `Sample: ${label}`;
            detailStatus.className = 'detail-val status-ready';
            previewBox.innerHTML = `<img src="${previewUrl}" alt="${label}">`;
            runBtn.disabled = false;

            // Automatically run classification on the selected sample
            await executeClassification();
          });
        });
      } else {
        samplesContainer.innerHTML = '<div class="samples-loading">No sample scenes found in dataset.</div>';
      }
    } catch (err) {
      console.warn('Failed to fetch /samples:', err);
      samplesContainer.innerHTML = `<div class="samples-loading">Unable to load sample library: ${err.message}</div>`;
    }

    // Action button listeners
    runBtn.addEventListener('click', executeClassification);
    resetBtn.addEventListener('click', resetState);
  }

  // Initialize on DOM ready
  document.addEventListener('DOMContentLoaded', () => {
    initDragAndDrop();
    initializeApp();
  });
})();
