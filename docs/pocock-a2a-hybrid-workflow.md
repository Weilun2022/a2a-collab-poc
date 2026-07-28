# Pocock × A2A 混合協作模式

跨機器部署用的參考文件。任何一台已經裝好 [Matt Pocock skill chain](https://github.com/mattpocock/skills) 與本 repo 的 A2A 工具（`ask_gemini.ps1`）的環境，都可以照這份文件的規則運作。

## 前提

- 專案已跑過 `/setup-matt-pocock-skills`，`CLAUDE.md` 有 `## Agent skills` 區塊
- 本機已裝 A2A 工具：`C:\Users\<user>\.claude\tools\a2a-ask\ask_gemini.ps1`（來源：本 repo）

## 核心規則

在 `/grilling`（或 `/grill-with-docs`）、`/to-spec`、`/to-tickets` 這幾個關卡被使用者觸發後：

1. **改成 Claude↔GPT 自主討論收斂，不逐題問使用者。** Claude 仍然負責探索程式碼、蒐集事實、做出實際決策/取捨；GPT（透過 A2A）扮演對這些決策施壓的角色。「一問一答收斂」這個迴圈跑在 Claude 與 GPT 之間，不是 Claude 與使用者之間。
2. **使用者的 slash command 關卡不變。** Claude 仍然不能自己觸發 `/grilling`/`/to-spec`/`/to-tickets`/`/implement`——這些指令必須由使用者親自打。改變的只是「關卡打開之後、裡面怎麼跑」。
3. **遇到只有使用者才知道答案的業務/情境判斷**（不是 GPT 能幫忙解決的技術分歧），**用最合理的猜測繼續走，不中斷 loop**，但在最後總結裡明確標出「這點是用猜的」。真的卡死、猜不出合理答案才中斷詢問使用者。
4. **`/implement` 前，只給使用者看一次白話文總結**：涵蓋 grilling/spec/tickets 全程決議的關鍵決策、理由，以及哪些地方是用猜的（要明確標出，例如「這點我們用猜的，因為只有你知道實際業務情況：...」）。使用者看的是這一份總結，不是每個關卡的原始討論記錄。滿意後才打 `/implement`——這之後的實作、`/code-review` 流程完全不變。

**Why：** 使用者要的是自主討論迴圈的產出速度（不用每個微決策都停下來問人），同時保留 Pocock 流程原本的安全網（spec/tickets 可審閱、implement 要明確授權才開始、code-review 照跑）。A2A 提供真正獨立的第二個模型意見，讓決策不是 Claude 自己一個人錨定在第一個想法上，同時不需要使用者在中間過程投入注意力，只在真正該他判斷的時候（`/implement` 前的最終確認）才要他看。

## 已知的兩個優化點（來自實戰回顧）

### 1. `-Prompt` 參數含特殊字元/多段落時，預設改用 `-PromptFile`

`ask_gemini.ps1` 的 `-Prompt` 參數若帶中文引號、`<`/`>` 之類符號，會被 PowerShell 的參數解析搞爛，直接跳錯誤。工具本身的 help 文字其實已經寫了「長內容/特殊字元要用 `-PromptFile`」，但實務上容易撞了才想到要切換。

**規則：只要 prompt 有多段落、任何特殊符號、或超過一兩句，一律預設用 `-PromptFile`，不要等出錯才切換。** 做法是先把 prompt 寫進暫存檔（用 `Write` 工具，UTF-8 無 BOM），再用 `-PromptFile <路徑>` 呼叫，避免 PowerShell 參數解析這一層風險。

### 2. 對 A2A 下 prompt 要刻意要求「唱反調」，不要問「你同意嗎」

實戰回顧發現：一次 session 裡三次 A2A 諮詢，GPT 幾乎都是「同意 Claude 的判斷」，沒有一次主動抓到 Claude 沒想到的問題。反而真正抓到問題的兩次,都不是來自 A2A（一次是使用者當面糾正「該用真實驗證而不是猜」，一次是 code-review 的 Spec 軸 subagent 自己讀規格文件抓到違反 Out of Scope）。

**原因分析**：下給 GPT 的 prompt 本身傾向「你同意這個判斷嗎」這種誘導同意的問法，等於在找背書,不是真的在對抗性地找漏洞。

**規則：往後對 A2A 下 prompt，要刻意要求對方唱反調、專門找方案的破綻、抓可能沒想到的邊界情況**，而不是徵求同意。例如：

- ❌ 弱：「我打算用 XX 粒度做快取，你覺得合理嗎？」
- ✅ 好：「我打算用 XX 粒度做快取，理由是 YY。請扮演懷疑的資深工程師，専門挑這個方案的漏洞、列出可能被忽略的邊界情況（併發、失敗恢復、資料量成長等），不要只回覆同不同意。」

只有這樣 A2A 才能真的發揮「第二個獨立意見」的價值，而不是變成一個橡皮圖章。

## 什麼時候不適用

- 純技術性、範圍明確的單點決策（例如「快取粒度該用哪個」「票號要不要拆」）：A2A 回應快（幾十秒內），適合拿來壓力測試，實測結論後續都有在 M: 實機驗證/code-review 中得到驗證,沒出現過「聽 A2A 建議結果反而錯了」的狀況。
- 真正的實測/驗證類問題（例如「這個功能在真實瀏覽器裡到底跑不跑得動」）：A2A 沒有能力驗證,不該把它當作驗證手段的替代品,還是要實際跑。

## 參考

- A2A 工具本身：[[reference_openrouter_collab_tool]] 的後繼者,見本 repo 的 `ask_gemini.py`/`ask_gemini.ps1`
- Pocock 流程硬性規則：[[feedback_pocock_workflow_discipline]]
