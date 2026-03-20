# Smart Comment Classification (SCC) — Technical Documentation
## Part 3 of 3: Frontend Architecture, Data Flow, API Reference, Testing, Deployment & Appendices

---

> **Document Series**
> - Part 1: Executive Summary, System Architecture, Technology Stack, Backend Core (Sections 1–5D)
> - Part 2: Backend Pipeline Completion, ML Model Deep Dives, Fine-Tuning (Sections 5E–7)
> - **Part 3 (this document):** Frontend Architecture, Data Flow, Full API Reference, Testing Infrastructure, Deployment, Performance, Security, Limitations, Glossary (Sections 8–18)

---

## Table of Contents — Part 3

- [Section 8: Frontend Architecture Deep Dive](#section-8-frontend-architecture-deep-dive)
  - [8.1 Component Hierarchy and Module Boundaries](#81-component-hierarchy-and-module-boundaries)
  - [8.2 App.jsx — Root State Machine](#82-appjsx--root-state-machine)
  - [8.3 NavBar Component](#83-navbar-component)
  - [8.4 TextInput Component](#84-textinput-component)
  - [8.5 FileUpload Component](#85-fileupload-component)
  - [8.6 SingleResult Component and anime.js Timeline](#86-singleresult-component-and-animejs-timeline)
  - [8.7 BatchResults Component](#87-batchresults-component)
  - [8.8 Footer Component](#88-footer-component)
  - [8.9 Theme System and CSS Architecture](#89-theme-system-and-css-architecture)
- [Section 9: Data Flow and Sequence Diagrams](#section-9-data-flow-and-sequence-diagrams)
  - [9.1 Single Text Classification Flow](#91-single-text-classification-flow)
  - [9.2 Batch File Classification Flow](#92-batch-file-classification-flow)
  - [9.3 Backend Internal Classification Sequence](#93-backend-internal-classification-sequence)
- [Section 10: Complete StarUML Class Diagrams](#section-10-complete-staruml-class-diagrams)
  - [10.1 Backend Core Class Diagram](#101-backend-core-class-diagram)
  - [10.2 Frontend React Component Class Diagram](#102-frontend-react-component-class-diagram)
  - [10.3 Training Pipeline Class Diagram](#103-training-pipeline-class-diagram)
  - [10.4 API Request and Response Data Model Diagram](#104-api-request-and-response-data-model-diagram)
  - [10.5 Model Registry and Status Architecture Diagram](#105-model-registry-and-status-architecture-diagram)
- [Section 11: Full API Reference](#section-11-full-api-reference)
  - [11.1 GET /health](#111-get-health)
  - [11.2 POST /classify/text](#112-post-classifytext)
  - [11.3 POST /classify/file](#113-post-classifyfile)
  - [11.4 GET /classify/status/{job_id}](#114-get-classifystatusjob_id)
  - [11.5 Error Response Schema](#115-error-response-schema)
  - [11.6 Rate Limiting Headers](#116-rate-limiting-headers)
- [Section 12: File Ingestion and Parsing](#section-12-file-ingestion-and-parsing)
  - [12.1 Supported Formats](#121-supported-formats)
  - [12.2 Multi-Column CSV Negotiation Protocol](#122-multi-column-csv-negotiation-protocol)
  - [12.3 Row Sanitisation](#123-row-sanitisation)
  - [12.4 Export Pipeline](#124-export-pipeline)
- [Section 13: Testing Infrastructure](#section-13-testing-infrastructure)
  - [13.1 Test Architecture Overview](#131-test-architecture-overview)
  - [13.2 FakeTokenizer Mock](#132-faketokenizer-mock)
  - [13.3 FakeBatchClassifier Mock](#133-fakebatchclassifier-mock)
  - [13.4 Fake Model Functions](#134-fake-model-functions)
  - [13.5 Test Case Breakdown](#135-test-case-breakdown)
  - [13.6 Running the Test Suite](#136-running-the-test-suite)
  - [13.7 Test Coverage Gaps and Recommendations](#137-test-coverage-gaps-and-recommendations)
- [Section 14: Deployment and Configuration](#section-14-deployment-and-configuration)
  - [14.1 Environment Variables](#141-environment-variables)
  - [14.2 Backend Local Development](#142-backend-local-development)
  - [14.3 Frontend Local Development](#143-frontend-local-development)
  - [14.4 Production Deployment Patterns](#144-production-deployment-patterns)
  - [14.5 Docker Considerations](#145-docker-considerations)
  - [14.6 GPU vs. CPU Execution](#146-gpu-vs-cpu-execution)
- [Section 15: Performance Analysis](#section-15-performance-analysis)
  - [15.1 Single-Text Latency Breakdown](#151-single-text-latency-breakdown)
  - [15.2 Batch Throughput Characteristics](#152-batch-throughput-characteristics)
  - [15.3 Memory Footprint](#153-memory-footprint)
  - [15.4 Frontend Performance](#154-frontend-performance)
- [Section 16: Security Considerations](#section-16-security-considerations)
  - [16.1 Input Validation](#161-input-validation)
  - [16.2 Rate Limiting Architecture](#162-rate-limiting-architecture)
  - [16.3 CORS Policy](#163-cors-policy)
  - [16.4 File Upload Security](#164-file-upload-security)
  - [16.5 Known Attack Surface](#165-known-attack-surface)
- [Section 17: Known Limitations and Future Work](#section-17-known-limitations-and-future-work)
  - [17.1 Current Limitations](#171-current-limitations)
  - [17.2 Roadmap Candidates](#172-roadmap-candidates)
- [Section 18: Glossary](#section-18-glossary)

---

## Section 8: Frontend Architecture Deep Dive

The SCC frontend is a single-page application (SPA) built with **React 19** and **Vite 6**. It communicates with the FastAPI backend over HTTP, renders classification results with rich animations, and supports both single-text and bulk-file workflows. The UI library of choice is **Mantine v7**, supplemented with **Lucide React** icons, **Framer Motion** (`motion/react`) for page-level transitions, and **anime.js** for granular element-level animations inside the result cards.

### 8.1 Component Hierarchy and Module Boundaries

```
App (App.jsx)
├── NavBar (NavBar.jsx)
│   ├── [Mantine] SegmentedControl  (Text / File mode toggle)
│   └── [Mantine] ActionIcon        (Theme toggle)
│
├── main#main-content  (two-column layout-col grid)
│   ├── Column 1 (input area)
│   │   ├── [mode=text] TextInput (TextInput.jsx)
│   │   └── [mode=file] FileUpload (FileUpload.jsx)   [lazy-loaded]
│   │       └── [mode=file, loading] Progress bar (inline JSX in App.jsx)
│   │
│   └── Column 2 (results area)
│       ├── [mode=text, result] SingleResult (SingleResult.jsx)
│       ├── [mode=file, results] BatchResults (BatchResults.jsx)  [lazy-loaded]
│       └── [empty states] inline motion.div placeholders
│
└── Footer (Footer.jsx)
```

The split into lazy-loaded chunks (`FileUpload` and `BatchResults`) is a conscious bundle-splitting decision. The text-input workflow, which is the primary entry point, loads immediately. The file upload machinery and the heavy Recharts dependency are deferred until the user explicitly switches to "File Upload" mode.

### 8.2 App.jsx — Root State Machine

`App.jsx` is the application controller. All cross-cutting state lives here; child components receive only what they need via props. The state variables and their roles are:

| State variable | Type | Purpose |
|---|---|---|
| `mode` | `'text' \| 'file'` | Which input panel is active |
| `theme` | `'dark' \| 'light'` | Current colour scheme; persisted to `localStorage` under the key `scc-theme` |
| `loading` | `boolean` | Whether an API call is in flight; drives loading spinners |
| `error` | `string` | Last error message; auto-clears after 5,000 ms via `window.setTimeout` |
| `singleResult` | `object \| null` | Full `/classify/text` response payload |
| `batchResults` | `array \| null` | Array of classified rows returned by the job poller |
| `batchFile` | `File \| null` | Reference to the original `File` object, used to render the file name in `BatchResults` |
| `progress` | `0–100` | Percentage complete for a running batch job |
| `runtimeLabel` | `string` | Display name of the active sentiment model, fetched from `/health` |
| `backendStatus` | `'checking' \| 'ok' \| 'degraded' \| 'offline'` | Backend health; drives the status badge in `NavBar` |

**Health polling.** On mount, a single `axios.get` call to `${API_URL}/health` runs. The response is parsed to extract `active_sentiment_display_name`, `sentiment_display_name`, or `display_name` (in order of preference) to produce a human-readable runtime label. The `isMounted` guard prevents state updates after the component unmounts.

**Error auto-dismissal.** Every time the `error` state changes to a non-empty string, a `useEffect` registers a 5-second timer. If the error changes again before the timer fires, the old timer is cleared and a new one is registered.

**`pollJobStatus(jobId)`.** After a successful file upload the backend returns a `job_id`. The front end calls `window.setInterval` at a 1,000 ms interval. On each tick it calls `GET /classify/status/{jobId}`. When `data.status === 'processing'` it updates the progress bar as `Math.round((processed / total) * 100)`. When `data.status === 'done'` it sets `batchResults`, clears the interval, and sets `loading = false`. On any fetch error the interval is also cleared and an error message is shown.

**Theme persistence.** Changing the theme writes to `localStorage` and to `document.documentElement.dataset.theme`. All CSS custom properties are scoped to `[data-theme="dark"]` and `[data-theme="light"]` attribute selectors, so the entire app re-themes without a page reload.

**MantineProvider.** The root `<MantineProvider>` is given a custom theme created with `createTheme()`. The `fontFamily` maps to `var(--font-primary)`, a CSS variable defined in `App.css`. The primary colour is a ten-shade `blue` ramp. `forceColorScheme={theme}` synchronises Mantine's internal dark/light logic with the SCC theme state so that all Mantine components (Buttons, Selects, Alerts) respect the same theme token.

### 8.3 NavBar Component

`NavBar.jsx` renders the top navigation bar and receives four props: `mode`, `onModeChange`, `theme`, `onThemeToggle`, and `backendStatus`.

**Backend status badge.** The `backendStatus` prop is translated into a label string (`'Online'`, `'Degraded'`, `'Offline'`, `'Checking'`) and a Mantine `Badge` colour (`green`, `yellow`, `gray`). This badge gives operators an instant visual indicator without opening the browser's network tab.

**Mode toggle.** A Mantine `SegmentedControl` with values `'text'` and `'file'` drives the mode. When the value changes, `handleModeToggle` fires an anime.js micro-animation (`scale: [1, 0.985, 1]`) on the wrapper `div` to give the toggle a subtle tactile feel, then calls `onModeChange`.

**Theme toggle.** A Mantine `ActionIcon` wraps a `motion.span` that carries the sun/moon icon. The icons cross-fade using `AnimatePresence mode="wait"` with a `rotate + scale + opacity` transition when the key (`theme`) changes. Additionally, clicking the button triggers a scale-bounce + rotate animation via anime.js directly on the `ActionIcon` DOM node, layering two animation systems on the same element for complementary effects.

### 8.4 TextInput Component

`TextInput.jsx` renders a textarea and a classify button. It is a controlled component: the textarea value is managed internally, while the classification action is delegated to the `onClassify` prop supplied by `App`.

Key behaviours:
- **Character counter**: displays the current character count as the user types, providing feedback before hitting the model's token limit.
- **Clear button**: appears when `hasResult` is `true`, allowing the user to reset the result panel without clearing the textarea.
- **Loading state**: the classify button receives `loading={loading}` from `App` and disables itself and shows a spinner while an API call is in flight.
- **Keyboard submit**: pressing `Ctrl+Enter` (or `Cmd+Enter` on macOS) submits the form, matching common text editor conventions.
- **Empty guard**: the classify button is disabled when the textarea is empty or contains only whitespace, preventing spurious API calls.

### 8.5 FileUpload Component

`FileUpload.jsx` is a drag-and-drop file selector with an embedded two-phase column selection flow. It uses Mantine's `Paper`, `Button`, `Select`, `Badge`, and `Text` components alongside Lucide icons.

**File validation.** The `validateFile` helper checks two constraints before any state is updated:
1. Extension must be one of `.csv`, `.txt`, `.xlsx` (enforced by comparing the lower-cased file extension to the `ACCEPTED` array).
2. File size must not exceed 10 MB (`MAX_SIZE = 10 * 1024 * 1024`).

Failures surface as an inline `role="alert"` error message within the component rather than relying on the global error toast in `App.jsx`.

**Drag-and-drop implementation.** Three event handlers are attached to the drop zone `<div>`:
- `onDrop`: calls `event.preventDefault()` to suppress the browser's default file-open behaviour, then extracts `event.dataTransfer.files[0]` and passes it to `handleFile`.
- `onDragOver`: calls `event.preventDefault()` (required to make the element a valid drop target) and sets `dragging = true`, which applies a `dragging` CSS class to highlight the zone.
- `onDragLeave`: clears `dragging`.

**Accessibility.** When no file is selected, the drop zone is rendered with `role="button"` and `tabIndex={0}`, making it keyboard-focusable. The `handleKeyDown` handler fires a click on the hidden `<input type="file" ref={inputRef}>` when `Enter` or `Space` is pressed, enabling keyboard-only file selection.

**Two-phase column negotiation.** When the backend cannot determine which column contains the text to classify (multi-column CSV files with no obvious `comment` or `text` column), it returns `{ status: "needs_column", columns: [...] }`. The `FileUpload` component detects this in its `handleClassify` callback and sets `needsColumn = true` along with the available `columns` array. A Mantine `Select` dropdown appears, pre-selected to the first column. The user picks a column and clicks "Classify File" again, this time with `selectedColumn` populated. The backend then proceeds with the specified column.

**File chip.** Once a file is selected, the drop zone transitions to a compact "chip" layout showing the file name and human-readable size (`formatSize` converts bytes to B/KB/MB as appropriate). A dismiss button (`X` icon) calls `removeFile`, which clears all state and resets the hidden input via `inputRef.current.value = ''`.

### 8.6 SingleResult Component and anime.js Timeline

`SingleResult.jsx` is the most animation-heavy component in the application. It uses `useRef` to hold a reference to the card's root DOM node, and `useEffect` to schedule an anime.js timeline whenever the `result` prop changes.

**Animation timeline construction:**

```javascript
const timeline = anime.timeline({ easing: 'easeOutExpo' });

timeline
  .add({ targets: root, opacity: [0,1], translateY: [14,0], duration: 320 })
  .add({ targets: stagedNodes, opacity: [0,1], translateY: [12,0],
         duration: 420, delay: anime.stagger(55) }, '-=120')
  .add({ targets: bars, scaleX: [0,1], duration: 560,
         delay: anime.stagger(45), easing: 'easeOutQuart' }, '-=240');
```

The timeline has three stages:
1. **Card entrance** (320 ms): the entire card fades in from `opacity: 0` and slides up from `translateY: 14px`.
2. **Content stagger** (420 ms, overlapping by 120 ms): each of the queried elements (`.result-heading`, `.result-flags`, `.result-main`, `.confidence-title`, `.interactive-text-box`, `.word-summary-stats`, `.toxicity-bar-container`, `.confidence-trio`) animates in with a 55 ms stagger between each element.
3. **Confidence bar fill** (560 ms, overlapping by 240 ms): `.confidence-bar-fill` and `.conf-bar-fill` elements animate from `scaleX: 0` to `scaleX: 1`. Since they have `transform-origin: left`, this creates a left-to-right filling bar effect.

The cleanup function returned by `useEffect` calls `anime.remove()` on all targets to prevent animation conflicts when `result` changes rapidly (e.g., the user submits multiple comments in quick succession).

**Data rendering.** The component destructures a rich response object:
- `sentiment` / `label`: the top-level classification (`Positive`, `Neutral`, `Negative`)
- `sentiment_confidence`: object with `positive`, `neutral`, `negative` float values (0–1)
- `latency_ms`: end-to-end server-side processing time in milliseconds
- `comment_type`: classification into one of the nine heuristic categories
- `is_toxic` / `toxicity`: boolean flag and float score
- `word_analysis`: array of `{ text, sentiment }` objects for lexical breakdown
- `word_counts`: `{ total, positive, neutral, negative }` integer counts
- `is_uncertain`: boolean, true when max confidence falls below the 0.55 threshold
- `is_sarcastic` / `sarcasm_score`: boolean and float
- `is_english`: boolean, false when the text is detected as non-English

**Flag pills.** Three conditional badges render above the main result when applicable: "Low confidence" (uncertain), "Sarcasm detected (X%)" with a percentage, and "Non-English text". These are rendered only when their conditions are true, keeping the UI uncluttered for the common case.

**Word highlight rendering.** The `word_analysis` array is mapped to a series of `<span>` elements. Whitespace tokens are rendered as plain spans. All other tokens get a `word-highlight` class plus a sentiment-specific colour class (`highlight-positive`, `highlight-neutral`, `highlight-negative`). A `title` attribute provides the raw text and sentiment on hover for accessibility.

**Confidence trio.** Three horizontal bars represent the positive, neutral, and negative confidence percentages. Each bar is a `<div>` with a percentage-width inline style, animated via the `scaleX` timeline stage described above.

### 8.7 BatchResults Component

`BatchResults.jsx` renders the analytics dashboard and paginated table after a batch job completes. It receives the `results` array (one object per input row) and the `originalFile` reference from `App`.

**useMemo-heavy data processing.** All derived data is memoised:
- `filteredData`: filters `results` by `filter` (All/Positive/Neutral/Negative) and `search` substring match on the comment text. On filter or search change, `page` is reset to 1.
- `paginatedData`: slices `filteredData` into pages of 20 rows.
- `stats`: counts per-label totals and converts them to percentages for the pie chart legend.
- `advancedStats`: counts `is_toxic`, `is_uncertain`, and `is_sarcastic` flags, and finds the most common `comment_type` by frequency sorting the `byType` accumulator.

**Recharts PieChart.** The donut chart uses `innerRadius={30}` and `outerRadius={45}`, with `paddingAngle={5}` between segments and `stroke="none"` to remove default white borders. `Cell` components map each segment to its corresponding CSS variable colour (`var(--positive)`, `var(--neutral)`, `var(--negative)`), ensuring the chart respects the active theme. The `Tooltip` is styled with inline `contentStyle` and `itemStyle` to match the card background and text colour.

**KPI grid.** Four `<article>` elements display:
1. Top Comment Type (most frequent `comment_type` in the batch)
2. Toxic Rows (count of rows where `is_toxic === true`)
3. Low Confidence (count where `is_uncertain === true`)
4. Sarcasm Flags (count where `is_sarcastic === true`)

**Backwards compatibility.** The table rendering code handles two result formats: the "new format" (when `row.sentiment !== undefined`, produced by the full pipeline) and the "old format" (legacy output with `row.label`, `row.confidence_positive`, etc.). The `isNewFormat` boolean gates which property names are accessed. This ensures exported CSVs from older backend versions can still be displayed.

**Export to CSV.** The `handleExport` function uses a **dynamic import** (`await import('xlsx')`) to load the `xlsx` library only when the user requests an export, avoiding the bundle cost for users who never export. The `XLSX.utils.json_to_sheet` method transforms the results array into a worksheet, and `XLSX.writeFile` triggers a browser download with a filename in `classified_results_YYYYMMDD.csv` format.

**Pagination controls.** A `ChevronLeft`/`ChevronRight` pair of buttons step through pages. The controls are hidden when `totalPages <= 1`. The current page and total are displayed as "Page N of M".

### 8.8 Footer Component

`Footer.jsx` is a stateless presentational component that receives `runtimeLabel` and `backendStatus` as props and renders a thin footer bar. It displays the model name and a colour-coded status dot. This is a secondary indicator redundant with the NavBar badge, providing persistent context at the bottom of the page for users who have scrolled past the navigation.

### 8.9 Theme System and CSS Architecture

The SCC frontend uses a layered CSS architecture:

**CSS custom properties (design tokens).** Root-level tokens are declared in `App.css` under two data-theme selectors:
```css
[data-theme="dark"] {
  --bg-base: #0e0f12;
  --bg-card: #161820;
  --text-primary: #e8eaf0;
  --positive: #34d399;
  --neutral: #60a5fa;
  --negative: #f87171;
  /* ... */
}

[data-theme="light"] {
  --bg-base: #f5f6fa;
  --bg-card: #ffffff;
  --text-primary: #1a1c23;
  /* ... */
}
```

All colour references in component CSS files use these variables. Switching the `data-theme` attribute on `<html>` causes an instant cascade update across all components.

**Glass card utility.** The `.glass-card` class applies a backdrop blur, semi-transparent background, and subtle border using `backdrop-filter: blur(12px)` and `background: var(--bg-card-glass)`. This class is used on `SingleResult`, `BatchResults`, `FileUpload`, and the progress container.

**Animation utility classes.** `animate-fade-in` and `animate-slide-in` are defined in `App.css` as single-use CSS keyframe animations. They provide entry animations for components that do not use anime.js or Framer Motion.

**Component-scoped CSS.** Each component has a corresponding `.css` file (e.g., `SingleResult.css`, `BatchResults.css`). This prevents style leakage while avoiding the overhead of CSS-in-JS.

---

## Section 9: Data Flow and Sequence Diagrams

### 9.1 Single Text Classification Flow

```
User types text + clicks "Classify"
        |
        v
[App.jsx] handleClassifyText(text)
  - setLoading(true)
  - setError('')
  - setBatchResults(null)
  - setProgress(0)
        |
        v
axios.post(`${API_URL}/classify/text`, { text })
        |
        |---HTTP POST /classify/text--->  [FastAPI main.py]
        |                                  rate_limit_check(request)
        |                                  classify_text_endpoint(body)
        |                                    preprocess_text(text)
        |                                    is_gibberish(processed)
        |                                    detect_language(processed)
        |                                    split_sentences(processed)
        |                                    classify_texts_internal([processed])
        |                                      _run_stage_batch("sentiment", [text])
        |                                      _run_stage_batch("toxicity", [text])
        |                                      _run_stage_batch("type", [text])
        |                                      _run_stage_batch("emotion", [text])
        |                                      _run_stage_batch("sarcasm", [text])
        |                                    assemble_result(outputs)
        |                                    _apply_sarcasm_adjustment(result)
        |                                    apply_type_heuristics(result)
        |                                    vader_word_analysis(original_text)
        |                                    compute_confidence_flags(result)
        |<--JSON response (ClassifyResponse)--
        |
[App.jsx]
  - setSingleResult(response.data)
  - setLoading(false)
        |
        v
[SingleResult.jsx] receives result prop
  - anime.js timeline fires
  - word highlights render
  - confidence bars animate
```

### 9.2 Batch File Classification Flow

```
User selects file + clicks "Classify File"
        |
        v
[FileUpload.jsx] handleClassify()
  → calls App.handleClassifyFile(file, selectedColumn)
        |
        v
[App.jsx] handleClassifyFile(file, column)
  - setLoading(true)
  - FormData: append(file), append(column)
        |
        v
axios.post(`${API_URL}/classify/file`, formData)
        |
        |---HTTP POST /classify/file---> [FastAPI main.py]
        |                                  parse_file_upload(file)
        |                                  if columns > 1 and no column specified:
        |                                    return { status: "needs_column", columns: [...] }
        |                                  else:
        |                                    schedule background task
        |                                    jobs[job_id] = { status: "processing", total: N, ... }
        |                                    BackgroundTask: process_batch_job(job_id, rows)
        |                                    return { job_id: "uuid", status: "processing" }
        |<--{ job_id, status: "processing" }--
        |
[App.jsx]
  - if status === "needs_column": surface column selector in FileUpload
  - else: pollJobStatus(job_id)
        |
        v
setInterval(1000ms):
  axios.get(`${API_URL}/classify/status/${jobId}`)
        |
        |---GET /classify/status/{id}---> [FastAPI]
        |                                   jobs[id] → { status, processed, total, results }
        |<--{ status: "processing", processed: N, total: M }--
        |  (or { status: "done", results: [...] })
        |
[App.jsx]
  if processing: setProgress(pct)
  if done:
    - setBatchResults(data.results)
    - setProgress(100)
    - setLoading(false)
    - clearInterval
        |
        v
[BatchResults.jsx] receives results prop
  - PieChart renders
  - KPI grid renders
  - Paginated table renders
```

### 9.3 Backend Internal Classification Sequence

```
classify_texts_internal([text1, text2, ...textN])
        |
        +---> preprocess_text(textI)  [for each text]
        |       - CONTRACTIONS expand
        |       - SLANG_MAP expand
        |       - COMMON_TYPOS correct
        |       - hashtag strip
        |       - URL remove
        |       - emoji replace
        |       - repeated char normalise
        |       - whitespace collapse
        |
        +---> _run_stage_batch("sentiment", processed_texts)
        |       - truncate_for_model(text, "sentiment")  [each text]
        |       - sentiment_classifier(batch)  [true batch call]
        |       - normalise labels to Positive/Neutral/Negative
        |       - return per-text scores
        |
        +---> _run_stage_batch("toxicity", processed_texts)  [parallel concept]
        |
        +---> _run_stage_batch("type", processed_texts)
        |       - BART zero-shot with 6 candidate labels
        |       - top-1 label extracted
        |
        +---> _run_stage_batch("emotion", processed_texts)
        |       - top-1 GoEmotions label
        |
        +---> _run_stage_batch("sarcasm", processed_texts)
        |       - irony/non_irony scores
        |
        +---> [for each text]:
                assemble base result dict
                _apply_sarcasm_adjustment(result)
                apply_type_heuristics(result)  [9-rule heuristic engine]
                vader_word_analysis(original_text)
                flag is_uncertain (max_conf < 0.55)
                multi-sentence aggregation if applicable
```

---

## Section 10: Complete StarUML Class Diagrams

The following diagrams use a StarUML-compatible textual notation. Each class box shows its name, attributes (`-` for private, `+` for public), and methods with parameter lists. Associations and dependencies are noted after the class definitions.

### 10.1 Backend Core Class Diagram

```
+-------------------------------------------------------+
|                    FastAPI App                        |
|-------------------------------------------------------|
| + app: FastAPI                                        |
| + jobs: Dict[str, JobRecord]                          |
| + model_registry: Dict[str, Pipeline]                 |
| + model_status: Dict[str, ModelStatusRecord]          |
| + model_tokenizers: Dict[str, AutoTokenizer]          |
| + rate_limit_store: Dict[str, Deque[float]]           |
|-------------------------------------------------------|
| + lifespan(app): asynccontextmanager                  |
| + load_model(): None                                  |
| + health(): HealthResponse                            |
| + classify_text_endpoint(body): ClassifyResponse      |
| + classify_file_endpoint(file, col, bg): JobResponse  |
| + classify_status(job_id): StatusResponse             |
+-------------------------------------------------------+

+-------------------------------------------------------+
|                  ClassifyRequest                      |
|-------------------------------------------------------|
| + text: str                                           |
+-------------------------------------------------------+

+-------------------------------------------------------+
|                  ClassifyResponse                     |
|-------------------------------------------------------|
| + sentiment: str                                      |
| + comment_type: str                                   |
| + is_toxic: bool                                      |
| + toxicity: float                                     |
| + is_uncertain: bool                                  |
| + is_sarcastic: bool                                  |
| + sarcasm_score: float                                |
| + is_english: bool                                    |
| + sentiment_confidence: SentimentConfidence           |
| + word_analysis: List[WordAnalysisEntry]              |
| + word_counts: WordCounts                             |
| + latency_ms: float                                   |
| + heuristics_applied: List[str]                       |
+-------------------------------------------------------+

+-------------------------------------------------------+
|               SentimentConfidence                     |
|-------------------------------------------------------|
| + positive: float                                     |
| + neutral: float                                      |
| + negative: float                                     |
+-------------------------------------------------------+

+-------------------------------------------------------+
|               WordAnalysisEntry                       |
|-------------------------------------------------------|
| + text: str                                           |
| + sentiment: str  // Positive|Neutral|Negative        |
|                   // |Whitespace                      |
+-------------------------------------------------------+

+-------------------------------------------------------+
|                   WordCounts                          |
|-------------------------------------------------------|
| + total: int                                          |
| + positive: int                                       |
| + neutral: int                                        |
| + negative: int                                       |
+-------------------------------------------------------+

+-------------------------------------------------------+
|                  JobRecord (TypedDict)                |
|-------------------------------------------------------|
| + status: str   // processing|done|error              |
| + total: int                                          |
| + processed: int                                      |
| + results: List[Dict]                                 |
| + error: Optional[str]                                |
+-------------------------------------------------------+

+-------------------------------------------------------+
|                ModelStatusRecord                      |
|-------------------------------------------------------|
| + loaded: bool                                        |
| + required: bool                                      |
| + model: str                                          |
| + display_name: str                                   |
| + max_tokens: int                                     |
| + error: Optional[str]                                |
| + attempted_models: List[str]                         |
+-------------------------------------------------------+

+-------------------------------------------------------+
|                  ModelSpec (Dict)                     |
|-------------------------------------------------------|
| + candidates: List[ModelCandidate]                    |
| + task: str                                           |
| + max_tokens: int                                     |
| + required: bool                                      |
| + display_name: str                                   |
+-------------------------------------------------------+

+-------------------------------------------------------+
|                ModelCandidate (Dict)                  |
|-------------------------------------------------------|
| + model: str                                          |
| + label_map: Optional[Dict[str,str]]                  |
+-------------------------------------------------------+

Associations:
  FastAPI App "uses" ClassifyRequest (POST /classify/text)
  FastAPI App "produces" ClassifyResponse
  FastAPI App "stores" JobRecord (1..*)
  FastAPI App "holds" ModelStatusRecord (1..5)
  ClassifyResponse "contains" SentimentConfidence (1)
  ClassifyResponse "contains" WordAnalysisEntry (0..*)
  ClassifyResponse "contains" WordCounts (1)
  ModelSpec "contains" ModelCandidate (1..*)
```

### 10.2 Frontend React Component Class Diagram

```
+----------------------------------------------------------+
|                      App                                 |
|----------------------------------------------------------|
| - mode: 'text'|'file'                                    |
| - theme: 'dark'|'light'                                  |
| - loading: boolean                                       |
| - error: string                                          |
| - singleResult: object|null                              |
| - batchResults: array|null                               |
| - batchFile: File|null                                   |
| - progress: number                                       |
| - runtimeLabel: string                                   |
| - backendStatus: string                                  |
|----------------------------------------------------------|
| + handleClassifyText(text): Promise<void>                |
| + handleClassifyFile(file, col): Promise<object>         |
| + pollJobStatus(jobId): void                             |
| + handleModeChange(mode): void                           |
| + handleClear(): void                                    |
| + toggleTheme(): void                                    |
+----------------------------------------------------------+

+----------------------------------------------------------+
|                      NavBar                              |
|----------------------------------------------------------|
| # themeToggleRef: Ref<HTMLElement>                       |
| # modeToggleRef: Ref<HTMLElement>                        |
|  [props] mode, onModeChange, theme, onThemeToggle,       |
|          backendStatus                                   |
|----------------------------------------------------------|
| + handleThemeClick(): void                               |
| + handleModeToggle(nextMode): void                       |
+----------------------------------------------------------+

+----------------------------------------------------------+
|                    TextInput                             |
|----------------------------------------------------------|
| - text: string                                           |
|  [props] onClassify, loading, hasResult, onClear         |
|----------------------------------------------------------|
| + handleSubmit(): void                                   |
| + handleKeyDown(event): void                             |
+----------------------------------------------------------+

+----------------------------------------------------------+
|                   FileUpload                             |
|----------------------------------------------------------|
| - file: File|null                                        |
| - error: string                                          |
| - dragging: boolean                                      |
| - columns: string[]                                      |
| - selectedColumn: string                                 |
| - needsColumn: boolean                                   |
| # inputRef: Ref<HTMLInputElement>                        |
|  [props] onClassify, loading                             |
|----------------------------------------------------------|
| + validateFile(file): boolean                            |
| + handleFile(file): void                                 |
| + handleDrop(event): void                                |
| + handleDragOver(event): void                            |
| + handleDragLeave(): void                                |
| + handleKeyDown(event): void                             |
| + removeFile(): void                                     |
| + handleClassify(): Promise<void>                        |
| + formatSize(bytes): string                              |
+----------------------------------------------------------+

+----------------------------------------------------------+
|                  SingleResult                            |
|----------------------------------------------------------|
| # cardRef: Ref<HTMLDivElement>                           |
|  [props] result, runtimeLabel                            |
|----------------------------------------------------------|
| + [useEffect] setupAnimeTimeline(): cleanup              |
| + getWordClass(sentiment): string                        |
+----------------------------------------------------------+

+----------------------------------------------------------+
|                  BatchResults                            |
|----------------------------------------------------------|
| - page: number                                           |
| - search: string                                         |
| - filter: string                                         |
|  [props] results, originalFile                           |
|  [memo] filteredData, paginatedData, stats, advancedStats|
|----------------------------------------------------------|
| + handleExport(): Promise<void>                          |
| + getLabelIcon(label): JSX                               |
+----------------------------------------------------------+

Associations:
  App "renders" NavBar (1)
  App "renders" TextInput (0..1, mode=text)
  App "renders" FileUpload (0..1, mode=file, lazy)
  App "renders" SingleResult (0..1, mode=text && result)
  App "renders" BatchResults (0..1, mode=file && results, lazy)
  App "renders" Footer (1)
```

### 10.3 Training Pipeline Class Diagram

```
+----------------------------------------------------------+
|              EncodedSentimentDataset                     |
|  (inherits torch.utils.data.Dataset)                     |
|----------------------------------------------------------|
| + encodings: Dict[str, Any]                              |
| + labels: List[int]                                      |
|----------------------------------------------------------|
| + __len__(): int                                         |
| + __getitem__(index): Dict[str, Tensor]                  |
+----------------------------------------------------------+

+----------------------------------------------------------+
|               TrainingPipelineModule                     |
|  (conceptual grouping of module-level functions)         |
|----------------------------------------------------------|
| + LABEL_TO_ID: Dict[str, int]                            |
| + ID_TO_LABEL: Dict[int, str]                            |
| + DEFAULT_BASE_MODEL: str                                |
| + DEFAULT_OUTPUT_DIR: Path                               |
| + TWEET_EVAL_LABELS: Dict[int, str]                      |
|----------------------------------------------------------|
| + parse_args(): Namespace                                |
| + load_table(path): DataFrame                            |
| + normalize_label(label): str                            |
| + compute_macro_f1(preds, labels): float                 |
| + build_examples(df, text_col, label_col): List[Dict]    |
| + build_examples_from_hf(records, ...): List[Dict]       |
| + make_dataset(tokenizer, examples, maxlen):             |
|     EncodedSentimentDataset                              |
| + split_examples(examples, ratio, seed):                 |
|     Tuple[List, List]                                    |
| + load_examples(args): Tuple[List, List]                 |
| + main(): None                                           |
+----------------------------------------------------------+

Associations:
  TrainingPipelineModule "creates" EncodedSentimentDataset (2 per run: train + eval)
  TrainingPipelineModule "wraps" HuggingFace Trainer (1)
  TrainingPipelineModule "wraps" AutoModelForSequenceClassification (1)
  TrainingPipelineModule "wraps" AutoTokenizer (1)
```

### 10.4 API Request and Response Data Model Diagram

```
HTTP Layer
+-------------------+      POST /classify/text      +---------------------+
|  ClassifyRequest  |----------------------------->  |  ClassifyResponse   |
|-------------------|                                |---------------------|
| text: str         |                                | sentiment: str      |
+-------------------+                                | comment_type: str   |
                                                     | is_toxic: bool      |
+-------------------+      POST /classify/file       | toxicity: float     |
|  FileUploadForm   |----------------------------->  | is_uncertain: bool  |
|-------------------|      (multipart/form-data)     | is_sarcastic: bool  |
| file: UploadFile  |                                | sarcasm_score: float|
| column?: str      |  --> 1. needs_column response  | is_english: bool    |
+-------------------+  --> 2. JobResponse            | sentiment_confidence|
                                                     | word_analysis: []   |
+-------------------+      GET /classify/status/{id} | word_counts: {}     |
|  StatusRequest    |----------------------------->  | latency_ms: float   |
|-------------------|                                | heuristics_applied  |
| job_id: str (path)|      +--------------------+   +---------------------+
+-------------------+      |   StatusResponse   |
                           |--------------------|
                           | status: str        |
                           | processed: int     |
                           | total: int         |
                           | results: List[Dict]|
                           | error?: str        |
                           +--------------------+

+------------------+       GET /health             +-------------------+
| (no body)        |----------------------------->  |  HealthResponse   |
+------------------+                               |-------------------|
                                                   | status: str       |
                                                   | model_status: {}  |
                                                   | active_sentiment_ |
                                                   |   model: str      |
                                                   | active_sentiment_ |
                                                   |   display_name:str|
                                                   +-------------------+
```

### 10.5 Model Registry and Status Architecture Diagram

```
MODEL_SPECS (global Dict)
+----------------------------------------------------------+
|  "sentiment":                                            |
|    candidates: [                                         |
|      { model: "answerdotai/ModernBERT-base",             |
|        label_map: {...} },                               |
|      { model: "cardiffnlp/twitter-roberta-base-          |
|                sentiment-latest",                        |
|        label_map: {...} }                                |
|    ]                                                     |
|    task: "text-classification"                           |
|    max_tokens: 512                                       |
|    required: true                                        |
|    display_name: "ModernBERT Sentiment"                  |
|                                                          |
|  "toxicity":                                             |
|    candidates: [{ model: "unitary/toxic-bert" }]         |
|    task: "text-classification"                           |
|    max_tokens: 512                                       |
|    required: false                                       |
|                                                          |
|  "type":                                                 |
|    candidates: [{ model: "facebook/bart-large-mnli" }]   |
|    task: "zero-shot-classification"                      |
|    max_tokens: 1024                                       |
|    required: false                                       |
|                                                          |
|  "emotion":                                              |
|    candidates: [                                         |
|      { model: "SamLowe/roberta-base-go_emotions" }       |
|    ]                                                     |
|    task: "text-classification"                           |
|    max_tokens: 512                                       |
|    required: false                                       |
|                                                          |
|  "sarcasm":                                              |
|    candidates: [                                         |
|      { model: "cardiffnlp/twitter-roberta-base-irony" }  |
|    ]                                                     |
|    task: "text-classification"                           |
|    max_tokens: 512                                       |
|    required: false                                       |
+----------------------------------------------------------+
                    |
                    | load_model() iterates MODEL_SPECS
                    v
model_registry (global Dict)
+----------------------------------------------------------+
|  "sentiment" -> Pipeline (loaded or None)                |
|  "toxicity"  -> Pipeline (loaded or None)                |
|  "type"      -> Pipeline (loaded or None)                |
|  "emotion"   -> Pipeline (loaded or None)                |
|  "sarcasm"   -> Pipeline (loaded or None)                |
+----------------------------------------------------------+
                    |
                    v
model_status (global Dict)
+----------------------------------------------------------+
|  "sentiment" -> { loaded: bool, error: str|None, ... }   |
|  "toxicity"  -> { ... }                                  |
|  "type"      -> { ... }                                  |
|  "emotion"   -> { ... }                                  |
|  "sarcasm"   -> { ... }                                  |
+----------------------------------------------------------+
```

---

## Section 11: Full API Reference

The backend exposes four HTTP endpoints. All responses use `Content-Type: application/json`. All error responses follow a uniform `{ "detail": "..." }` schema compatible with FastAPI's default `HTTPException` format.

### 11.1 GET /health

Returns the current system health and model loading status.

**Request:** No body required.

**Response 200 OK:**
```json
{
  "status": "ok",
  "model_status": {
    "sentiment": {
      "loaded": true,
      "required": true,
      "model": "answerdotai/ModernBERT-base",
      "display_name": "ModernBERT Sentiment",
      "max_tokens": 512,
      "error": null,
      "attempted_models": ["answerdotai/ModernBERT-base"]
    },
    "toxicity": { "loaded": true, ... },
    "type": { "loaded": true, ... },
    "emotion": { "loaded": false, "error": "OOM", ... },
    "sarcasm": { "loaded": true, ... }
  },
  "active_sentiment_model": "answerdotai/ModernBERT-base",
  "active_sentiment_display_name": "ModernBERT Sentiment"
}
```

When a required model fails to load, `status` will be `"degraded"` and the corresponding `model_status` entry will have `"loaded": false` with an `"error"` string.

**Status codes:** `200` always (health check never returns 5xx).

### 11.2 POST /classify/text

Classifies a single text comment.

**Request body:**
```json
{
  "text": "string — the comment to classify"
}
```
- `text` is required and must be a non-empty string after stripping whitespace.
- Maximum effective length: 512 tokens (the model's context window). Longer inputs are truncated server-side.

**Response 200 OK:**
```json
{
  "sentiment": "Positive",
  "comment_type": "Praise",
  "is_toxic": false,
  "toxicity": 0.02,
  "is_uncertain": false,
  "is_sarcastic": false,
  "sarcasm_score": 0.04,
  "is_english": true,
  "sentiment_confidence": {
    "positive": 0.91,
    "neutral": 0.06,
    "negative": 0.03
  },
  "word_analysis": [
    { "text": "This", "sentiment": "Neutral" },
    { "text": " ", "sentiment": "Whitespace" },
    { "text": "app", "sentiment": "Neutral" },
    { "text": " ", "sentiment": "Whitespace" },
    { "text": "is", "sentiment": "Neutral" },
    { "text": " ", "sentiment": "Whitespace" },
    { "text": "amazing", "sentiment": "Positive" }
  ],
  "word_counts": {
    "total": 4,
    "positive": 1,
    "neutral": 3,
    "negative": 0
  },
  "latency_ms": 142.3,
  "heuristics_applied": ["emotion_praise_boost"]
}
```

**Error 400:** Returned when text is empty or contains only whitespace.
```json
{ "detail": "Text must not be empty." }
```

**Error 429:** Rate limit exceeded.
```json
{ "detail": "Rate limit exceeded. Try again in 60 seconds." }
```

**Error 503:** Sentiment model not loaded (required model failure).
```json
{ "detail": "Sentiment model is unavailable." }
```

### 11.3 POST /classify/file

Uploads a file for batch classification. Uses `multipart/form-data`.

**Request form fields:**
- `file` (required): Binary file content. Accepted formats: `.csv`, `.txt`, `.xlsx`. Maximum size: 10 MB.
- `column` (optional): Name of the text column to classify. If omitted and the file contains multiple columns, the endpoint will return a `needs_column` response.

**Response 200 OK — Job started:**
```json
{
  "job_id": "3f7a1b2c-9e4d-4c8f-b12a-0e5f3d7a9c1b",
  "status": "processing",
  "total": 500
}
```
The background classification job begins immediately. Poll `/classify/status/{job_id}` to track progress.

**Response 200 OK — Column selection required:**
```json
{
  "status": "needs_column",
  "columns": ["comment_text", "timestamp", "user_id", "rating"]
}
```
Resubmit the same file with the `column` parameter set to the desired column name.

**Error 400:** Unsupported file format, corrupt file, or empty file.
```json
{ "detail": "Unsupported file format. Please upload CSV, TXT, or XLSX." }
```

**Error 422:** Multipart form validation failure (Pydantic / FastAPI standard).

### 11.4 GET /classify/status/{job_id}

Polls the status of a background batch classification job.

**Path parameter:** `job_id` — UUID string returned by `/classify/file`.

**Response 200 OK — In progress:**
```json
{
  "status": "processing",
  "processed": 120,
  "total": 500,
  "results": []
}
```

**Response 200 OK — Complete:**
```json
{
  "status": "done",
  "processed": 500,
  "total": 500,
  "results": [
    {
      "comment": "Great product!",
      "sentiment": "Positive",
      "comment_type": "Praise",
      "is_toxic": false,
      "toxicity": 0.01,
      "is_uncertain": false,
      "is_sarcastic": false,
      "conf_pos": 0.93,
      "conf_neu": 0.05,
      "conf_neg": 0.02
    }
  ]
}
```
Note: Batch results use abbreviated confidence keys (`conf_pos`, `conf_neu`, `conf_neg`) to reduce response payload size compared to the single-text `sentiment_confidence` object.

**Error 404:** Job ID not found in the in-memory store.
```json
{ "detail": "Job not found." }
```

**Error 200 with `"status": "error"`:** Job failed during processing.
```json
{
  "status": "error",
  "processed": 85,
  "total": 500,
  "results": [],
  "error": "Model inference failed on row 85."
}
```

### 11.5 Error Response Schema

All errors follow FastAPI's standard `HTTPException` format:
```json
{
  "detail": "Human-readable error description."
}
```

Validation errors from Pydantic models return a structured `422 Unprocessable Entity` with a `detail` array describing each field validation failure.

### 11.6 Rate Limiting Headers

The current implementation does not inject custom rate-limit headers into responses. When the limit is hit, only `HTTP 429` with the detail message is returned. Future versions should add `Retry-After`, `X-RateLimit-Limit`, and `X-RateLimit-Remaining` headers.

---

## Section 12: File Ingestion and Parsing

### 12.1 Supported Formats

The backend accepts three file types for batch classification:

| Format | Extension | Parser | Notes |
|---|---|---|---|
| Comma-separated values | `.csv` | `pandas.read_csv` | Auto-detects delimiter via pandas default engine |
| Excel workbook | `.xlsx` | `pandas.read_excel(engine="openpyxl")` | First sheet only; requires `openpyxl` installed |
| Plain text | `.txt` | Custom line reader | Each non-empty line is one comment |

For `.txt` files, no column detection is needed — the entire file is treated as a single-column dataset with one comment per line. For `.csv` and `.xlsx` files, column auto-detection is attempted first.

### 12.2 Multi-Column CSV Negotiation Protocol

When a `.csv` or `.xlsx` file is uploaded without a `column` parameter, the backend:

1. Reads the file into a pandas `DataFrame`.
2. Inspects column names for common text-column keywords: `comment`, `text`, `review`, `content`, `message`, `body` (case-insensitive substring matching).
3. If exactly one column matches, it is selected automatically and the job proceeds.
4. If zero or multiple columns match, the backend returns `{ "status": "needs_column", "columns": <list_of_all_column_names> }` without starting a job.
5. The frontend surfaces the column selector UI (Mantine `Select`).
6. The user selects a column and resubmits with the `column` query parameter.
7. The backend re-reads the file with the specified column, validates it exists, and starts the background job.

This two-round protocol avoids requiring users to know the column name in advance, which is common when they are uploading CSV exports from third-party platforms (e.g., app store review exports, survey tools).

### 12.3 Row Sanitisation

Before each row is sent to the classification pipeline, it undergoes:
- Coercion to string via `str(value)` (handles numeric or NaN cells gracefully)
- Strip of leading/trailing whitespace
- Empty string check — empty rows are skipped and not included in the results, but the job's `total` counter reflects only non-empty rows

### 12.4 Export Pipeline

After batch classification completes, the user may export results as a CSV from the `BatchResults` component. The export process:

1. User clicks "Export CSV"; `handleExport` async function fires.
2. `import('xlsx')` — the `xlsx` package is dynamically imported (lazy), preventing it from increasing the initial JS bundle size.
3. `XLSX.utils.json_to_sheet(mappedRows)` transforms the results array into a worksheet object. Each row is projected to: `{ ID, Comment, Sentiment, Comment_Type, Is_Toxic, Confidence_Positive, Confidence_Neutral, Confidence_Negative, Toxicity_Score }`.
4. The new format (`row.sentiment !== undefined`) and legacy format are both handled via the `isNew` boolean guard.
5. `XLSX.writeFile(wb, filename, { bookType: 'csv' })` triggers a browser download. Despite using the `xlsx` library, the output format is CSV (plain comma-separated), not XLSX. This was chosen for maximum compatibility with downstream tools.
6. The filename includes a `YYYYMMDD` date stamp generated from `new Date()`.

---

## Section 13: Testing Infrastructure

### 13.1 Test Architecture Overview

The SCC backend test suite lives in `backend/tests/test_backend.py` and uses Python's built-in `unittest` framework together with FastAPI's `TestClient` (which wraps Starlette's `TestClient` using the `httpx` library).

The test strategy centres on **dependency injection via module-level variable substitution**. Because the backend stores all model pipelines in module-level global variables (`sentiment_classifier`, `toxicity_classifier`, `type_classifier`, `emotion_classifier`, `sarcasm_classifier`), the test `setUp` can directly replace these with lightweight mock objects. No monkey-patching framework is required.

The test class `BackendRegressionTests` inherits from `unittest.TestCase`. Six regression tests exercise distinct layers of the system:

1. Text preprocessing correctness
2. Tokenizer-aware truncation
3. Sarcasm detection and sentiment adjustment
4. Batch inference batching behaviour
5. Health endpoint response structure
6. Model candidate fallback chain integrity

### 13.2 FakeTokenizer Mock

```python
class FakeTokenizer:
    def __call__(self, text, truncation=False, max_length=None,
                 return_overflowing_tokens=False,
                 return_attention_mask=False):
        tokens = text.split()
        sliced = tokens[:max_length] if truncation and max_length else tokens
        overflow = (tokens[max_length:]
                    if truncation and max_length and len(tokens) > max_length
                    else [])
        payload = {"input_ids": sliced}
        if return_overflowing_tokens:
            payload["overflowing_tokens"] = overflow
        return payload

    def decode(self, token_ids, skip_special_tokens=True,
               clean_up_tokenization_spaces=True):
        return " ".join(token_ids)
```

`FakeTokenizer` simulates a HuggingFace tokenizer using whitespace-based word splitting as a proxy for subword tokenisation. This is sufficient for testing the truncation logic because the `truncate_for_model` function calls the tokenizer with `return_overflowing_tokens=True` and then uses `decode` to reconstruct the truncated string. By treating words as tokens, the mock makes it easy to predict exactly how many words will survive a given `max_length`.

### 13.3 FakeBatchClassifier Mock

```python
class FakeBatchClassifier:
    def __init__(self, fn):
        self.fn = fn
        self.batch_lengths = []

    def __call__(self, texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        self.batch_lengths.append(len(texts))
        results = [self.fn(text, **kwargs) for text in texts]
        return results if len(results) > 1 else results[0]
```

`FakeBatchClassifier` wraps a pure-function mock (`fn`) and records the sizes of all batches it receives in `batch_lengths`. This is used by `test_batch_inference_uses_real_batched_model_calls` to assert that a two-text input produces batch calls of size 2 to the sentiment and type classifiers, verifying that the backend's `_run_stage_batch` function does not fall back to single-item inference.

The callable returns a list when called with multiple texts and a single item when called with one, mirroring the HuggingFace pipeline's actual behaviour.

### 13.4 Fake Model Functions

Five pure functions serve as the prediction logic for each fake classifier:

| Function | Trigger keywords | Simulates |
|---|---|---|
| `fake_sentiment` | `"bad"`, `"crashed"`, `"terrible"` → Negative; `"great"`, `"love"`, `"amazing"` → Positive; else Neutral | Twitter RoBERTa / ModernBERT output format |
| `fake_sarcasm` | `"great"` AND `"crashed"` → irony score 0.91 | Irony model output |
| `fake_toxicity` | `"idiot"` → toxic score 0.77; else 0.04 | Toxic-BERT output |
| `fake_emotion` | `"thanks"` / `"love"` → gratitude; `"crashed"` → annoyance; else neutral | GoEmotions output |
| `fake_type` | `"?"` → question; `"buy now"` → spam; `"great"` / `"love"` → praise; else feedback | BART zero-shot output (dict with `labels`/`scores`) |

These functions return output structures that exactly match the real HuggingFace pipeline output format (list of `{"label": ..., "score": ...}` dicts for text-classification tasks; `{"labels": [...], "scores": [...]}` dict for zero-shot classification tasks).

### 13.5 Test Case Breakdown

**`test_preprocess_text_normalizes_slang_and_hashtags`**

Input: `"OMG #LoveThis sooo much!!!"`

Assertions:
- `"oh my god"` appears in the processed output (OMG slang expansion)
- `"Love This"` appears (hashtag camel-case splitting)
- `"sooo"` does not appear (repeated character normalisation reduces `sooo` to `so`)

This test validates the entire preprocessing chain in one shot, ensuring the three most common normalisation operations work together without interference.

**`test_tokenizer_truncation_is_token_aware`**

Temporarily sets `MODEL_SPECS["sentiment"]["max_tokens"] = 4`, then calls `truncate_for_model("one two three four five six", "sentiment")`.

Assertions:
- Returned prepared text is `"one two three four"` (first four whitespace-separated tokens)
- `meta["truncated"]` is `True`

This test guards against regression to character-based truncation, which would incorrectly split mid-word.

**`test_sarcasm_uses_non_top1_irony_score`**

Input: `"Oh great, crashed 5 times today"`

Because the fake sarcasm classifier returns irony score 0.91 for text containing both "great" and "crashed", and the fake sentiment classifier returns Positive for "great", the sarcasm adjustment logic should flip the label to Negative.

Assertions:
- `result["sentiment"] == "Negative"`
- `result["is_sarcastic"] == True`
- `result["sarcasm_score"] > 0.8`
- `"sarcasm_negative_context"` is in `result["heuristics_applied"]`

This is the most complex regression test. It validates that the three-gate sarcasm logic (high irony score + sentiment discordance + negative-context detection) works end-to-end.

**`test_batch_inference_uses_real_batched_model_calls`**

Input: a list of two texts.

Assertions:
- `len(results) == 2`
- `2 in sentiment_classifier.batch_lengths` (the two texts were batched into one call)
- `2 in type_classifier.batch_lengths`

This test is an architectural correctness test: it verifies that the `classify_texts_internal` function does not loop and call the model one text at a time.

**`test_health_endpoint_exposes_model_status`**

Uses FastAPI's `TestClient` to make a real HTTP request to `GET /health`.

Assertions:
- HTTP 200 status
- `payload["status"] == "ok"`
- `"model_status"` key present
- `"active_sentiment_model"` key present

This is the only test that exercises the HTTP layer, providing integration coverage for the health endpoint response structure.

**`test_sentiment_model_candidates_include_fallback`**

Calls `backend_main._resolve_model_candidates("sentiment")` directly.

Assertions:
- At least one candidate is returned
- The last candidate's `"model"` is `DEFAULT_SENTIMENT_FALLBACK_MODEL` (the Twitter RoBERTa model)

This test verifies that the fallback chain is correctly configured — if the primary ModernBERT model fails to load, the pipeline will not leave the sentiment model slot empty.

### 13.6 Running the Test Suite

```bash
# From the repository root
cd backend
python -m pytest tests/test_backend.py -v

# Or using unittest directly
python -m unittest tests.test_backend -v

# Expected output:
# test_batch_inference_uses_real_batched_model_calls ... ok
# test_health_endpoint_exposes_model_status ... ok
# test_preprocess_text_normalizes_slang_and_hashtags ... ok
# test_sarcasm_uses_non_top1_irony_score ... ok
# test_sentiment_model_candidates_include_fallback ... ok
# test_tokenizer_truncation_is_token_aware ... ok
# Ran 6 tests in 0.8s — OK
```

The tests run without GPU or internet access because all model weights are replaced by the fake classifiers. Average runtime is under 2 seconds on any modern machine.

### 13.7 Test Coverage Gaps and Recommendations

The current test suite provides valuable regression coverage for core logic, but several areas remain untested:

| Gap | Recommended test |
|---|---|
| `/classify/file` endpoint happy path | Upload a real CSV bytes object via `TestClient` |
| Rate limiter enforcement | Submit 61 requests in 60 seconds and assert the 61st returns 429 |
| `process_batch_job` background function | Call directly with a mock job record and assert `jobs[id]["status"] == "done"` |
| `apply_type_heuristics` all nine rules | Parameterised tests with inputs designed to trigger each rule |
| VADER word analysis output structure | Unit test `vader_word_analysis` independently |
| Language detection for non-English text | Input a known Spanish text and assert `is_english == False` |
| File format parsing | Unit test `load_table` for CSV, XLSX, and TXT inputs |
| Column auto-detection | Test `needs_column` branch with a multi-column CSV |
| Frontend component tests | Vitest + React Testing Library for `FileUpload` and `BatchResults` |

---

## Section 14: Deployment and Configuration

### 14.1 Environment Variables

**Backend:**

| Variable | Default | Description |
|---|---|---|
| `HOST` | `0.0.0.0` | Interface to bind the Uvicorn server |
| `PORT` | `8000` | TCP port for the backend HTTP server |
| `TRANSFORMERS_CACHE` | HuggingFace default | Cache directory for downloaded model weights |
| `HF_HOME` | HuggingFace default | Alternative HuggingFace home directory |
| `CUDA_VISIBLE_DEVICES` | (all GPUs) | Restrict GPU visibility; set to `""` to force CPU-only |

**Frontend:**

| Variable | Default | Description |
|---|---|---|
| `VITE_API_URL` | `http://localhost:8000` | Backend base URL. Override for production deployments. |

`VITE_API_URL` must be set at build time (it is inlined by Vite's static replacement). For dynamic runtime configuration in production, the value must be baked into the build.

### 14.2 Backend Local Development

**Prerequisites:**
- Python 3.10 or later
- CUDA 12.x (optional but strongly recommended for acceptable performance)

**Setup:**
```bash
cd backend
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# First run downloads model weights (~3–8 GB total)
python main.py
# or
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The `--reload` flag enables hot-reloading during development. Note that on first startup without cached weights, the model download phase can take 10–30 minutes depending on network speed.

**Key dependencies** (from `requirements.txt`):
```
fastapi==0.115.0
uvicorn==0.30.6
transformers>=4.48.0
torch>=2.4.1
pandas>=2.1.0
openpyxl>=3.1.2
langdetect>=1.0.9
vaderSentiment>=3.3.2
python-multipart>=0.0.9
```

### 14.3 Frontend Local Development

**Prerequisites:**
- Node.js 18 or later
- npm 9 or later

**Setup:**
```bash
cd frontend
npm install
VITE_API_URL=http://localhost:8000 npm run dev
# Vite dev server starts on http://localhost:5173
```

**Build for production:**
```bash
VITE_API_URL=https://your-api.example.com npm run build
# Output in frontend/dist/
```

**Preview production build:**
```bash
npm run preview
```

### 14.4 Production Deployment Patterns

**Pattern A: Single-server (co-located frontend and backend)**

The most common pattern for small deployments. Uvicorn serves the FastAPI backend on port 8000. Nginx sits in front, serving the built frontend static files and proxying `/api/*` or direct path requests to Uvicorn.

```nginx
server {
    listen 80;
    root /var/www/scc/frontend/dist;
    index index.html;

    location /classify {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location /health {
        proxy_pass http://127.0.0.1:8000;
    }

    location / {
        try_files $uri $uri/ /index.html;
    }
}
```

**Pattern B: Separate services**

Frontend deployed to a CDN/static host (Vercel, Netlify, Cloudflare Pages). Backend deployed to a compute instance (EC2, Cloud Run, Azure Container Apps). `VITE_API_URL` is set to the backend's public URL at build time.

**Pattern C: Docker Compose**

Two services: `backend` running `uvicorn main:app` and `frontend` running `npm run build && serve -s dist`. A shared `nginx` service acts as a reverse proxy. Model weights are mounted as a Docker volume to avoid re-downloading on container restarts.

### 14.5 Docker Considerations

**Image size.** The `transformers` and `torch` libraries are large. A naive `pip install -r requirements.txt` into a Python 3.11 slim image produces an image over 10 GB when model weights are included. Strategies:
- Mount model weights as an external volume; do not bake them into the image.
- Use `torch` CPU-only wheel (`torch==2.4.1+cpu`) to reduce from ~4 GB to ~1 GB for CPU-only deployments.
- Use multi-stage builds to keep the final image lean.

**Health check.** Configure Docker health checks to call `GET /health` every 30 seconds. The container should not receive traffic until the health endpoint returns `status: "ok"`, which confirms all required models have loaded.

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s \
  CMD curl -f http://localhost:8000/health || exit 1
```

The `--start-period=120s` accounts for the model loading time on first startup.

### 14.6 GPU vs. CPU Execution

The backend auto-detects GPU availability via `torch.cuda.is_available()`:
- **With GPU**: `fp16=True` is set in fine-tuning `TrainingArguments`. HuggingFace `pipeline()` defaults to `device=0`. Inference is 10–20x faster.
- **Without GPU**: `fp16=False`. All pipelines run on CPU. Single-text inference typically takes 2–8 seconds per request.

For production workloads with significant traffic, GPU execution is essentially required. A single NVIDIA A10 GPU (24 GB VRAM) can comfortably hold all five models simultaneously and serve hundreds of requests per minute.

---

## Section 15: Performance Analysis

### 15.1 Single-Text Latency Breakdown

End-to-end latency for a typical 15-word comment on a mid-range GPU (RTX 3070, 8 GB VRAM):

| Stage | Approximate time |
|---|---|
| HTTP receive + JSON parse | < 1 ms |
| Rate limit check (in-memory deque scan) | < 0.1 ms |
| `preprocess_text` (regex, dict lookups) | 0.5–2 ms |
| `truncate_for_model` × 5 models | 1–3 ms total |
| Sentiment model inference | 20–50 ms |
| Toxicity model inference | 15–40 ms |
| Type model inference (BART-large) | 80–200 ms |
| Emotion model inference | 15–40 ms |
| Sarcasm model inference | 15–40 ms |
| Heuristics + sarcasm adjustment | < 1 ms |
| VADER word analysis | 1–5 ms |
| JSON serialisation + HTTP send | < 2 ms |
| **Total (GPU)** | **~150–340 ms** |
| **Total (CPU)** | **~2,000–8,000 ms** |

The BART-large-MNLI type classifier dominates latency on GPU because it is a much larger model (406M parameters) than the others. Replacing it with a distilled model (e.g., `cross-encoder/nli-MiniLM2-L6-H768`) would reduce type classification latency to ~20 ms at some accuracy cost.

### 15.2 Batch Throughput Characteristics

The background batch processor calls `classify_texts_internal` in chunks. True model-level batching means that `N` texts are sent to each model pipeline as a single batch call rather than `N` individual calls. HuggingFace pipelines handle the internal padding and masking for variable-length inputs automatically.

Approximate throughputs on GPU:
- **ModernBERT sentiment** (batch size 8): ~400 texts/minute
- **BART-large type** (batch size 4): ~80 texts/minute (bottleneck)
- **Toxic-BERT** (batch size 8): ~350 texts/minute

Overall batch throughput is constrained by the BART type classifier at approximately 80 texts/minute on a single GPU.

### 15.3 Memory Footprint

All five models loaded simultaneously on GPU (approximate VRAM usage):

| Model | Parameters | VRAM (FP32) | VRAM (FP16) |
|---|---|---|---|
| ModernBERT-base | 149M | ~600 MB | ~300 MB |
| Twitter RoBERTa sentiment | 125M | ~500 MB | ~250 MB |
| Toxic-BERT | 110M | ~440 MB | ~220 MB |
| BART-large-MNLI | 406M | ~1,624 MB | ~812 MB |
| GoEmotions RoBERTa | 125M | ~500 MB | ~250 MB |
| Irony RoBERTa | 125M | ~500 MB | ~250 MB |
| **Total** | **~1,040M** | **~4.2 GB** | **~2.1 GB** |

On a consumer GPU with 8 GB VRAM, all models fit comfortably in FP16. On CPU, model weights are stored in system RAM. With 16 GB RAM, all models load without issues. With 8 GB RAM, BART-large may cause memory pressure.

### 15.4 Frontend Performance

**Bundle size.** The Vite production build produces:
- Main bundle: ~350–400 KB (gzipped: ~110 KB)
- Recharts chunk (lazy): ~200 KB (gzipped: ~60 KB)
- xlsx chunk (lazy, export only): ~500 KB (gzipped: ~150 KB)

The lazy splitting ensures the initial load does not include Recharts or xlsx, which are only needed for the batch results flow.

**Animation performance.** anime.js animations operate on individual DOM properties and use `requestAnimationFrame` internally. Framer Motion's `AnimatePresence` uses CSS transforms for transitions. Both libraries avoid layout-triggering properties (no `width`, `height`, or `top` animations on the critical path), keeping animations on the compositor thread for smooth 60 fps performance even on integrated graphics.

---

## Section 16: Security Considerations

### 16.1 Input Validation

**Backend text input.** `ClassifyRequest` validates that `text` is a non-empty string. Pydantic's type coercion handles unexpected input types (numbers, nulls) by converting them to strings or raising 422 errors.

**Injection safety.** No SQL or command execution is performed on the input text. The text passes through regex replacements and NLP model inference only. There is no risk of SQL injection or OS command injection in the classification pipeline.

**XSS via output.** The frontend renders user-supplied text from `word_analysis` entries. React's JSX renderer escapes HTML entities by default, so text containing `<script>` tags or event handlers is rendered as literal text, not executed.

### 16.2 Rate Limiting Architecture

The rate limiter is a sliding window counter implemented in memory:

```python
rate_limit_store: Dict[str, deque[float]] = defaultdict(deque)

def rate_limit_check(request: Request) -> None:
    ip = request.client.host
    now = time.monotonic()
    window = rate_limit_store[ip]

    # Remove timestamps older than 60 seconds
    while window and window[0] < now - 60:
        window.popleft()

    if len(window) >= 60:
        raise HTTPException(status_code=429, detail="Rate limit exceeded.")

    window.append(now)
```

**Limitations:**
- In-memory store is not shared across multiple worker processes (if running with `--workers > 1`).
- Not persistent — a server restart resets all rate limit counters.
- Does not account for load balancers that may present a single forwarded IP.

For production deployments, a Redis-backed rate limiter (e.g., `slowapi` with a Redis backend) is strongly recommended.

### 16.3 CORS Policy

The FastAPI application is configured with `CORSMiddleware`. In development, `allow_origins=["*"]` is commonly used. For production, this should be restricted to the specific frontend origin:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend.example.com"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
```

The application does not use cookies or authentication tokens, so `allow_credentials=False` is appropriate.

### 16.4 File Upload Security

**Size limit.** The 10 MB maximum is enforced both client-side (in `FileUpload.jsx`) and should be enforced server-side. FastAPI/Starlette does not apply a default upload size limit; a `ContentSizeLimitMiddleware` or Nginx `client_max_body_size` directive should be added in production.

**File type validation.** Only `.csv`, `.txt`, and `.xlsx` files are accepted. The validation is extension-based, not MIME-type-based. A malicious actor could rename a file to bypass the extension check. True file type validation would require inspecting the file magic bytes (e.g., via the `python-magic` library).

**Temporary file handling.** FastAPI's `UploadFile` stores uploaded content in memory for small files and in a temporary file for large ones. The application reads the content into a pandas DataFrame and does not write files to disk permanently. There is no risk of a path traversal attack from the upload itself.

### 16.5 Known Attack Surface

| Vector | Risk level | Mitigation |
|---|---|---|
| Rate limit bypass (IP spoofing via X-Forwarded-For) | Medium | Use `uvicorn --proxy-headers` and restrict trusted IPs |
| Denial of service via very long texts | Low | Tokenizer truncation caps effective input length |
| Denial of service via large file uploads | Medium | Add server-side size limit middleware |
| In-memory job store exhaustion | Low | Add TTL-based job eviction (currently jobs never expire) |
| Model inference manipulation (adversarial inputs) | Low | Out of scope for this application tier |

---

## Section 17: Known Limitations and Future Work

### 17.1 Current Limitations

**In-memory job store.** Background batch jobs are stored in a Python dictionary (`jobs: Dict[str, JobRecord]`). This means:
- Jobs are lost on server restart.
- The store grows unboundedly with no TTL eviction. In production, a server processing thousands of batch jobs per day would accumulate significant memory usage from completed job records.
- The store is not shared across multiple worker processes, so polling `/classify/status` from a different worker than the one that started the job will return 404.

**Single-worker architecture.** The backend is designed to run as a single Uvicorn worker. Multi-worker deployments (e.g., `uvicorn --workers 4`) would require a shared job store (Redis, PostgreSQL) and a shared model registry. Model weights cannot be loaded per-worker without significant VRAM usage multiplication.

**No authentication or authorization.** Any client with network access to the backend can classify text and upload files. For multi-tenant deployments or public-facing APIs, API key authentication or OAuth2 must be added.

**Language detection accuracy.** The `langdetect` library uses a probabilistic character n-gram model. For very short texts (under 10 words), detection accuracy drops significantly. A comment like "ok" may be misidentified as any language.

**Sarcasm detection in context.** The irony model is trained on Twitter data and performs best on short, informal texts. Longer, formal sarcasm (e.g., passive-aggressive corporate communication) may not be detected.

**VADER word analysis vs. model sentiment.** VADER and the neural sentiment model can disagree. The word highlight colours reflect VADER's lexicon-based judgment, while the overall sentiment label comes from the neural model. Users may find the visual breakdown "contradicting" the headline label in edge cases (e.g., a sentence where VADER sees many negative words but the model correctly identifies sarcasm and labels it Positive).

**No persistence or history.** There is no database, so there is no way to retrieve previous classifications, audit history, or aggregate analytics across sessions.

**Frontend-only error handling for network failures.** If the backend is reachable but a single model inference fails mid-pipeline, the error propagates as a generic 500 response. The frontend cannot distinguish between "model loaded but inference failed" and "model not loaded", limiting the ability to surface actionable error messages.

### 17.2 Roadmap Candidates

**High priority:**
- Redis-backed job store and rate limiter for multi-worker deployments
- Job TTL eviction (e.g., expire completed jobs after 1 hour)
- Server-side file upload size limit middleware
- API key authentication for public-facing deployments

**Medium priority:**
- Replace BART-large-MNLI type classifier with a distilled NLI model to reduce latency by 60–70%
- Add streaming endpoint (`/classify/stream`) for real-time character-by-character results using Server-Sent Events
- WebSocket-based batch status updates (replacing the current 1-second polling)
- Expand multi-language support: detect language and route to language-specific sentiment models
- Historical analytics dashboard: store classification results in SQLite/PostgreSQL and expose trend endpoints

**Low priority / nice-to-have:**
- Dark/light theme preference synced via `prefers-color-scheme` media query
- Keyboard shortcut sheet (accessible via `?` key)
- Configurable rate limit parameters via environment variables
- Export to Excel (XLSX with formatted headers and conditional colour formatting)
- CI/CD pipeline with GitHub Actions running the test suite on each commit
- Fine-tuned sarcasm model on domain-specific data (app store reviews vs. Twitter)

---

## Section 18: Glossary

**anime.js** — A lightweight JavaScript animation library that provides timeline-based sequencing, staggered delays, and per-target easing. Used in `SingleResult.jsx` and `NavBar.jsx` for element-level micro-animations.

**AnimatePresence** — A Framer Motion API that allows React components to animate when they are unmounted (exit animations). Used in `App.jsx` to crossfade between text-mode and file-mode views.

**BART (Bidirectional and Auto-Regressive Transformer)** — A Facebook AI sequence-to-sequence model pre-trained with denoising objectives. The `bart-large-mnli` variant is fine-tuned for Natural Language Inference, enabling zero-shot classification.

**Background task (FastAPI)** — FastAPI's `BackgroundTasks` mechanism schedules a Python coroutine to run after the HTTP response has been sent to the client. Used by `/classify/file` to process rows without blocking the HTTP response.

**Batch inference** — The practice of passing multiple inputs to a model in a single forward pass, exploiting GPU parallelism. SCC batches all texts for a given stage together before calling the pipeline.

**CORSMiddleware** — FastAPI middleware that adds Cross-Origin Resource Sharing headers to HTTP responses, allowing the frontend (served on a different origin) to make API calls to the backend.

**Confidence score** — The probability assigned by the model's softmax output to each class label. In SCC, three confidence scores are produced per classification (positive, neutral, negative), always summing to 1.0.

**Donut chart** — A variant of a pie chart with a hollow centre. Used in `BatchResults.jsx` via Recharts' `PieChart` with `innerRadius > 0`.

**Dynamic import** — A JavaScript syntax (`import()`) that loads a module asynchronously at runtime rather than at parse time. SCC uses this for the `xlsx` library to avoid including it in the initial bundle.

**FP16 (Half precision floating point)** — 16-bit floating point format. Neural network weights stored in FP16 occupy half the memory of FP32 with minimal accuracy loss for inference, enabling twice as many model parameters to fit in GPU VRAM.

**GeGLU (Gated Linear Unit with GELU activation)** — The feed-forward activation function used in ModernBERT, replacing the standard ReLU or GELU of earlier transformers. Provides richer non-linear expressivity.

**Glass card** — The visual design motif used for SCC's result cards. Achieved via `backdrop-filter: blur()` and semi-transparent backgrounds.

**GoEmotions** — A Google Research dataset of 58K Reddit comments annotated with 28 emotion categories, used to train the RoBERTa emotion classifier employed by SCC.

**Gradient accumulation** — A training technique that simulates a larger batch size by accumulating gradients over multiple forward passes before performing a weight update. Used in the ModernBERT fine-tuning script to train with effective batch sizes larger than what fits in VRAM.

**Heuristic boosting** — The process of modifying a model's predicted label or scores using hand-crafted rules. SCC's `apply_type_heuristics` function applies nine boosting rules to refine comment type labels based on emotion output, keyword detection, punctuation patterns, and other signals.

**Job record** — An in-memory dictionary entry created by the backend when a batch file upload job is submitted. Tracks status, progress, and results.

**Langdetect** — A Python port of Google's language detection library, which uses Naive Bayes classification over character n-gram profiles to identify the language of a text.

**Lazy loading (React)** — The use of `React.lazy()` and `Suspense` to defer the loading and rendering of a component until it is first needed. SCC uses lazy loading for `FileUpload` and `BatchResults`.

**Lifespan context manager** — An `asynccontextmanager` function passed to FastAPI's `lifespan` parameter. Code before the `yield` runs at startup; code after runs at shutdown. SCC uses it to load models at startup.

**Macro F1** — A classification metric that computes the F1 score independently for each class and averages them, giving equal weight to all classes regardless of class frequency. Used as the primary metric for ModernBERT fine-tuning.

**ModernBERT** — A 2024-vintage BERT-family encoder from AnswerDotAI featuring Rotary Positional Embeddings, FlashAttention 2, alternating local/global attention, GeGLU feed-forward layers, and an 8K token context window.

**MantineProvider** — The root context provider for the Mantine UI library. All Mantine components must be descendants of this provider to receive theme tokens.

**NLI (Natural Language Inference)** — The task of determining whether a premise entails, contradicts, or is neutral to a hypothesis. BART-MNLI uses NLI to implement zero-shot classification: the class labels become hypotheses and the input text becomes the premise.

**ONNX** — Open Neural Network Exchange, an open format for representing ML models. Not currently used in SCC but mentioned as a future optimisation path.

**ROPE (Rotary Position Embeddings)** — A positional encoding scheme that encodes position information via rotation matrices applied to query and key vectors in attention. Produces better length extrapolation than sinusoidal or learned absolute embeddings.

**Sarcasm adjustment** — The three-gate logic in `_apply_sarcasm_adjustment` that flips or moderates a sentiment label when all three conditions hold: high irony score, conflicting initial sentiment, and negative contextual keywords.

**Segmented control** — A UI widget where a single value is selected from a small set of mutually exclusive options, rendered as a group of adjacent buttons. Used in `NavBar` for Text / File mode selection.

**Sliding window rate limiter** — A rate limiter that tracks the exact timestamps of recent requests in a deque, removing entries older than the window duration on each check. Provides more accurate throttling than a fixed-window counter.

**Staggered animation** — An animation technique where multiple elements begin their animations with successive delays, creating a cascading visual effect. Implemented in SCC's `SingleResult` via `anime.stagger(55)`.

**VADER (Valence Aware Dictionary and sEntiment Reasoner)** — A rule-based sentiment analysis tool specifically designed for social media text. Uses a hand-curated lexicon of words with sentiment valence scores and a set of grammatical heuristics for negation, intensification, and punctuation.

**Vite** — A frontend build tool that uses native ES modules in development for instant hot-module replacement, and Rollup for production bundling. SCC uses Vite 6.

**Zero-shot classification** — Classifying inputs into categories that were not seen during model training, by reformulating the task as NLI. The BART-MNLI model in SCC classifies comment types without any fine-tuning on SCC-specific examples.

---

*End of Part 3. This document, together with Part 1 and Part 2, constitutes the complete SCC Technical Documentation set.*

---

**Document Metadata**

| Field | Value |
|---|---|
| Project | Smart Comment Classification (SCC) |
| Part | 3 of 3 |
| Sections covered | 8–18 |
| Total parts | 3 |
| Combined estimated pages | ~60 (at 500 words/page) |
| Revision date | 2026-03-20 |
| Primary technology stack | Python 3.11, FastAPI 0.115, React 19, Vite 6, Mantine 7, HuggingFace Transformers, PyTorch 2.4, anime.js, Framer Motion |
