# 專案工作流程 (Project Workflow)

本文件旨在詳細說明「AI 文本偵測器」專案從啟動 (Kick-off) 到完成結案的完整作業程序。專案全程遵循以 OpenSpec 為核心的規格驅動開發 (Spec-Driven Development) 模式，以確保開發過程的透明度、一致性與高品質。

---

## 階段一：專案啟動與規劃 (Project Kick-off & Planning)

此階段的目標是確立專案方向、定義核心需求，並建立一個所有參與者共同認可的開發藍圖。

1.  **需求定義 (Requirement Definition)**:
    *   與使用者（您）進行初步溝通，理解專案的核心目標：建立一個能夠區分 AI 與人類撰寫文本的工具。
    *   確認期望的技術棧（Python, scikit-learn, Streamlit）和最終產出形式（互動式 Web 應用程式）。

2.  **建立 OpenSpec 提案 (Creating the OpenSpec Proposal)**:
    *   基於定義好的需求，建立一個 OpenSpec 變更提案 (`001-implement-ai-text-detector`)。
    *   此提案包含一系列結構化文件，用以指導後續的開發工作：
        *   `proposal.md`: 說明此變更的**理由 (Why)**、**內容 (What)** 和**影響 (Impact)**。
        *   `design.md`: 闡述高層次的技術決策、架構模式與潛在風險。
        *   `spec.md`: 以明確、可驗證的需求 (Requirements) 和情境 (Scenarios) 形式，定義功能的具體行為。
        *   `tasks.md`: 將整個開發過程拆解為一個詳細、可追蹤的任務清單。

3.  **驗證與批准 (Validation & Approval)**:
    *   執行 `openspec validate` 指令，確保所有提案文件都符合 OpenSpec 的語法與結構規範。
    *   在所有文件準備就緒且通過驗證後，等待使用者（您）的明確批准，才會進入下一階段的實作。

---

## 階段二：迭代開發與實作 (Iterative Development & Implementation)

此階段是將規劃轉化為實際程式碼的核心過程，採用模組化、可測試且持續整合使用者回饋的迭代方式進行。

1.  **遵循任務清單 (Following the Task List)**:
    *   嚴格按照 `tasks.md` 中定義的順序，逐一完成每個開發任務。

2.  **模組化開發 (Modular Development)**:
    *   將應用程式的核心邏輯拆分為位於 `src/` 目錄下的多個獨立模組，以實現高內聚、低耦合的設計：
        *   `data_loader.py`: 負責資料載入。
        *   `feature_engineering.py`: 負責特徵提取。
        *   `model.py`: 負責模型訓練與預測。
        *   `visualization.py`: 負責繪製所有視覺化圖表。

3.  **單元測試 (Unit Testing)**:
    *   針對核心的後端邏輯模組（如 `data_loader` 和 `feature_engineering`），使用 `pytest` 框架編寫單元測試，確保其功能的正確性與穩定性。

4.  **介面開發與整合 (UI Development & Integration)**:
    *   開發主應用程式檔案 (`5114050013_hw5.py`)，建立 Streamlit 使用者介面。
    *   將後端模組的功能（如模型預測、圖表生成）逐步整合到前端介面中，實現完整的應用功能。

5.  **使用者回饋與修正 (User Feedback & Refinement)**:
    *   在每個主要功能整合完畢後，都會請求使用者進行驗證測試。
    *   根據使用者提出的回饋（例如 UI/UX 調整、功能優化、錯誤報告），對程式碼進行即時的迭代修正，確保最終產出符合使用者期望。

---

## 階段三：部署準備與文件化 (Deployment Preparation & Documentation)

在核心功能完成後，此階段專注於讓應用程式準備好部署，並完善所有相關文件。

1.  **準備部署版本 (Preparing for Deployment)**:
    *   分析並解決在雲端部署時可能遇到的問題（如 Streamlit Cloud 的短暫檔案系統）。
    *   做出關鍵決策，例如將「模型訓練」轉為離線步驟，並將**預先訓練好的模型 (`model.joblib`)** 打包進專案中，以確保部署後的穩定性。

2.  **撰寫與同步專案文件 (Writing & Synchronizing Documentation)**:
    *   基於最終的程式架構與邏輯，撰寫一份詳細的 `README.md`，包含專案介紹、CRISP-DM 框架、技術棧、執行步驟、部署指南等。
    *   回頭審視並更新所有 `.md` 專案文件（包括 `tasks.md`, `project.md` 等），確保其內容與最終成品完全一致。

3.  **推送至版本控制 (Pushing to Version Control)**:
    *   在每個開發階段或重大修改後，都會透過 `git` 將所有變更（包括程式碼和文件）推送到遠端的 GitHub 儲存庫，保持版本同步。

---

## 階段四：部署與結案 (Deployment & Closing)

此階段是專案的最後一步，完成雲端部署、最終測試，並正式結束本次開發週期。

1.  **雲端部署 (Cloud Deployment)**:
    *   由使用者（您）依照 `README.md` 中的指南，手動操作將應用程式部署到 Streamlit Cloud。

2.  **端對端測試 (End-to-End Testing)**:
    *   由使用者（您）訪問部署在雲端的公開網址，對所有功能進行最終的線上驗證。

3.  **歸檔變更 (Archiving the Change)**:
    *   在所有任務完成且通過最終測試後，執行 `openspec archive` 指令。
    *   此指令會將 `001-implement-ai-text-detector` 這個變更正式封存，並更新主規格文件，象徵本次開發週期的圓滿結束。

4.  **最終提交 (Final Commit)**:
    *   將歸檔操作所產生的檔案變更，連同其他所有最終修改，做一次性的提交與推送，確保 GitHub 儲存庫反映的是專案的最終、已歸檔狀態。
