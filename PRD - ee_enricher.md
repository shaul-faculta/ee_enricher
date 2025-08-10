## **1\. Overview & Purpose**

**Objective**: Enable users to enrich a CSV dataset of soil sampling locations (with `pointid`, `lat`, `lon`, plus any custom columns) by fetching relevant geospatial variables from Google Earth Engine (GEE).

**Key Features**:

* CSV import and flexible column mapping for `pointid`, latitude, and longitude.

* Dynamic user selection of Earth Engine layers via simple search.

* Batch data retrieval using Python Earth Engine (or similar) libraries.

* Seamless authentication via a configured Google service account—users are not required to authenticate manually.

* Built-in handling for GEE rate limits and quotas, including batching logic.

* Real-time progress feedback during enrichment.

---

## **2\. User Flow / Functional Requirements**

1. **CSV Upload & Column Mapping**

   * User uploads a CSV.

   * Tool previews sample rows and prompts the user to confirm which columns correspond to `pointid`, `lat`, and `lon`.

2. **Layer Selection Interface**

   * User can search and select from available Earth Engine layers (e.g., TerraClimate, SST, NDVI).

   * Option to specify variables or temporal date ranges.

3. **Authentication Setup (Admin Side)**

   * Administrator (tool owner) configures:

     * A Google Cloud project registered with Earth Engine.

     * Creation of a service account with appropriate roles (e.g., Earth Engine Resource Viewer/Writer, Storage roles if needed), and JSON key setup.

     * **UPDATED**: Service account must be registered in an Earth Engine-enabled project and have required IAM roles including Earth Engine Resource Viewer and Service Usage Consumer roles.

   * JSON credentials stored securely, perhaps via environment variables or secrets management.

   * Tool initializes Earth Engine using those credentials so end users aren't required to authenticate.

4. **Batch Fetch Execution**

   * Internal batching mechanism (e.g., default `batch_size=500`) to respect GEE's per-request limits and prevent memory/timeouts.

   * Users can optionally adjust batch size.

   * Leverage similar logic as your reference snippet: use `ee.ImageCollection`, compositing selected layers, and `reduceRegions()` for efficient lookup.

5. **Progress & Feedback**

   * Real-time progress bar indicating batch progression, estimated time remaining, success/fail alerts.

6. **Output Delivery**

   * Enriched table downloadable as CSV (or other formats: Excel, JSON).

   * Option to preview the first few rows within the tool.

---

## **3\. Technical Architecture & Considerations**

* **Backend Layer**:

  * Python-based processing leveraging the `earthengine-api` (`ee`) library.

  * **UPDATED AUTHENTICATION**: Use `ee.ServiceAccountCredentials()` and `ee.Initialize()` with JSON key and project ID. Support both environment variable patterns (`GOOGLE_APPLICATION_CREDENTIALS`) and direct JSON key file paths for flexibility.

  * Service account must be registered in an Earth Engine-enabled project and have required IAM roles.

  * **UPDATED**: Support for multiple authentication modes including service accounts for unattended code execution.

* **Rate Limits & Resource Management**:

  * **UPDATED QUOTAS**: Respect GEE's current limits including:
    - Adjustable limits: concurrent requests, request rate, batch tasks, asset storage
    - Fixed limits: computation time, per-request memory, aggregation constraints, table import size, request payload size, task queue length
    - **UPDATED**: Automatic retry process through Earth Engine client library for quota exceeded errors
    - **UPDATED**: Request caching, exponential backoff, and efficient algorithm design for optimization

  * Implement batching, error retries, and fallback strategies (e.g., split large jobs into smaller time/date ranges).

* **UI/Frontend Layer**:

  * Intuitive CSV uploader with auto-detection and manual override of column mapping.

  * Layer search UI integrating Earth Engine's catalog or a curated list.

  * Progress bar with batch number, rows processed, estimated time, and error logs.

  * Result preview and download.

---

## **4\. Non-Functional Requirements**

* **Usability**: Simple interface requiring minimal technical knowledge.

* **Security**: Secure handling of service account keys; no exposure to end users; audit logging of accesses.

* **Scalability**: Handle large point datasets via batching and queueing.

* **Reliability**: Graceful handling of network/timeouts; retries; clear error reporting.

* **Maintainability**: Modular codebase separating auth, data retrieval, batching, and UI.

---

## **5\. Admin & Deployment Workflow**

1. Register Cloud Project with Earth Engine and enable Earth Engine API.

2. Create service account and grant roles (e.g., Earth Engine Resource Viewer, Storage Viewer/Creator if exporting).

3. Generate and securely store JSON key.

4. Deploy tool with credentials configured (e.g., via env vars or secure vault).

5. Optionally rotate credentials and manage access logs.

---

## **6\. Milestones / Development Phases**

| Phase | Description |
| ----- | ----- |
| 1: Prototype | CSV import, column mapping, sample GEE retrieval using service account auth. |
| 2: Layer Selector & Batching | UI for layer search \+ implement batching logic and progress tracking. |
| 3: Quota Handling | Integrate GEE limit guards (batch size cap, retries, error detection). |
| 4: Output & UX | Table preview, downloads, UI polish. |
| 5: Security & Admin Setup | Secure key handling, service account setup documents, demos. |
| 6: Testing & Documentation | Unit/integration tests, admin instructions, user docs. |

---

## **7\. Risks & Mitigations**

* **Quota Exceeded / Memory Limits**  
   *Mitigation*: Use batching, limit concurrent aggregations, and provide guidance to users. **UPDATED**: Leverage Earth Engine client library's automatic retry process and implement request caching strategies.

* **Authentication Errors or Key Misconfig**  
   *Mitigation*: Validate service account at setup; test initialization on deployment. Provide clear admin troubleshooting steps. **UPDATED**: Ensure proper IAM role assignment including Earth Engine Resource Viewer and Service Usage Consumer roles.

* **Exporting Data to Wrong Drive (Service Account vs User)**  
   *Mitigation*: Be aware that service account exports (e.g., to Drive) go to service account's drive unless impersonation or proper drive access is managed.

---

## **8\. References**

* **Service Account Authentication Guide**: Steps to create, register, and initialize with service account credentials in Python.

* **Earth Engine Quotas & Batch Limits**: **UPDATED**: Current quota system includes adjustable project-level limits and fixed system-wide limits for computation time, memory usage, and request constraints.

* **Common Pitfalls**: Register service account for Earth Engine; ensure proper IAM role assignment; drive export behavior considerations.

* **UPDATED DOCUMENTATION DATES**: 
  - Service Account Guide: Last updated 2025-01-30 UTC
  - Authentication Guide: Last updated 2025-01-09 UTC  
  - Usage & Quotas Guide: Last updated 2025-07-07 UTC

