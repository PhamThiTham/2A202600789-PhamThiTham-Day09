**Họ tên:** Phạm Thị Thắm
**MSHV:** 2A202600789
**Lớp:** E403

---

# Phần 1: Direct LLM Calling

## 1. LLM được khởi tạo như thế nào?

LLM được khởi tạo thông qua hàm `get_llm()` trong file `common/llm.py`.

```python
def get_llm() -> ChatOpenAI:
    """Return a ChatOpenAI client pointed at OpenRouter."""
    return ChatOpenAI(
        model=os.getenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-4-5"),
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
        max_tokens=1024,
        temperature=0,
    )
```

Giải thích:

* `model`: chọn model từ biến môi trường.
* `openai_api_key`: API key để truy cập OpenRouter.
* `openai_api_base`: endpoint API của OpenRouter.
* `max_tokens`: giới hạn số token output.
* `temperature`: điều khiển độ ngẫu nhiên của câu trả lời.

---

## 2. Message được gửi đến LLM có cấu trúc gì?

```python
messages = [
    SystemMessage(
        content=(
            "You are a legal expert. Provide a clear, concise analysis "
            "of the legal question asked. Keep your response under 300 words."
        )
    ),
    HumanMessage(content=QUESTION),
]
```

Cấu trúc gồm:

* `SystemMessage`
* `HumanMessage`

Sau đó được gửi tới LLM bằng:

```python
response = await llm.ainvoke(messages)
```

---

## 3. Tại sao cần có `SystemMessage` và `HumanMessage`?

### `SystemMessage`

Dùng để:

* định nghĩa vai trò của AI,
* đặt quy tắc hoạt động,
* kiểm soát phong cách trả lời.

Ví dụ:

* yêu cầu AI đóng vai chuyên gia pháp lý,
* trả lời ngắn gọn dưới 300 từ.

### `HumanMessage`

Đại diện cho:

* câu hỏi của người dùng,
* nội dung chính mà model cần xử lý.

---

# Bài Tập 1.1 — Thay đổi câu hỏi

## Câu hỏi mới

```python
QUESTION = "Doanh nghiệp có thể bị xử phạt như thế nào nếu vi phạm hợp đồng lao động?"
```

## Kết quả chạy chương trình
```python
======================================================================
STAGE 1: Direct LLM Calling
======================================================================

[How it works]
  1. We send a system prompt + user question directly to the LLM
  2. The LLM responds from its training data only
  3. No tools, no retrieval, no external knowledge

Question: Doanh nghiệp có thể bị xử phạt như thế nào nếu vi phạm hợp đồng lao động?
----------------------------------------------------------------------

>>> Calling LLM directly (no tools, no RAG)...

# XỬ PHẠT VI PHẠM HỢP ĐỒNG LAO ĐỘNG ĐỐI VỚI DOANH NGHIỆP

## 1. Các hình thức xử phạt hành chính

**Mức phạt tiền:**
- **5-10 triệu đồng**: Vi phạm về nội dung hợp đồng, không giao hợp đồng cho người lao động
- **10-20 triệu đồng**: Không ký hợp đồng lao động khi đủ điều kiện
- **20-30 triệu đồng**: Ép buộc ký hợp đồng trái pháp luật
- **30-40 triệu đồng**: Vi phạm nghiêm trọng về thời giờ làm việc, nghỉ ngơi

**Biện pháp khắc phục:**
- Buộc ký hợp đồng lao động
- Buộc trả lương, phụ cấp chưa thanh toán
- Buộc bồi thường thiệt hại

## 2. Trách nhiệm dân sự

- **Bồi thường thiệt hại** cho người lao động theo quy định
- **Trả lương, phụ cấp** còn thiếu
- **Chi phí đào tạo** nếu chấm dứt hợp đồng trái luật

## 3. Trách nhiệm hình sự

Trong trường hợp đặc biệt nghiêm trọng:
- Cưỡng ép lao động
- Chiếm đoạt tiền lương
- Gây thiệt hại lớn cho người lao động

## 4. Hậu quả khác

- **Uy tín doanh nghiệp** bị ảnh hưởng
- **Khó tuyển dụng** nhân sự
- **Thanh tra lao động** tăng cường giám sát
- **Đình chỉ hoạt động** trong trường hợp vi phạm nghiêm trọng


Doanh nghiệp cần tuân thủ đúng Bộ luật Lao động 2019 để tránh các hình thức xử phạt trên, đồng thời xây dựng môi trường làm việc ổn định và bền vững.

----------------------------------------------------------------------
[Limitations of Stage 1]
  - Stateless: no conversation memory between calls
  - No tools: cannot search databases or calculate damages
  - Knowledge cutoff: only knows what was in training data
  - No grounding: cannot cite specific statutes or current case law

Next: Stage 2 adds RAG and tools to ground responses in real data.
======================================================================
```

Chương trình đã trả lời đúng theo ngữ cảnh pháp lý, bao gồm:

* xử phạt hành chính,
* trách nhiệm dân sự,
* trách nhiệm hình sự,
* hậu quả đối với doanh nghiệp.

Điều này cho thấy mô hình có thể tạo phản hồi hợp lý chỉ bằng direct prompting mà chưa cần RAG hay tools.

---

# Bài Tập 1.2 — Thêm Temperature Control

Đã chỉnh sửa `common/llm.py`:

```python
def get_llm() -> ChatOpenAI:
    """Return a ChatOpenAI client pointed at OpenRouter."""
    return ChatOpenAI(
        model=os.getenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-4-5"),
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
        max_tokens=1024,
        temperature=0.3,
    )
```

## Ý nghĩa

`temperature=0.3` giúp:

* output ổn định hơn,
* giảm độ ngẫu nhiên,
* phù hợp với các ứng dụng pháp lý cần tính chính xác và nhất quán cao.

## Phần 2: LLM + RAG & Tools (30 phút)

## 1. Hàm `@tool` decorator được dùng ở đâu?

`@tool` decorator được dùng để biến các hàm Python thành tools mà LLM có thể gọi.

Trong file `stages/stage_2_rag_tools/main.py`, `@tool` được dùng tại:

```python
@tool
def search_legal_database(query: str) -> str:
```

và:

```python
@tool
def calculate_damages(breach_type: str, contract_value: float) -> str:
```

Sau khi được decorate bằng `@tool`, các hàm này có thể:

* được bind vào LLM,
* được model tự động gọi,
* nhận arguments từ LLM.

---

## 2. `LEGAL_KNOWLEDGE` được cấu trúc như thế nào?

`LEGAL_KNOWLEDGE` là một list gồm nhiều dictionary.

Mỗi entry có cấu trúc:

```python
{
    "id": "...",
    "keywords": [...],
    "text": "..."
}
```

Ý nghĩa:

* `id`: định danh của knowledge entry,
* `keywords`: các từ khóa dùng để matching,
* `text`: nội dung kiến thức pháp lý.

Ví dụ:

```python
{
    "id": "nda_trade_secret",
    "keywords": ["nda", "non-disclosure", "confidential"],
    "text": "NDA breaches may trigger..."
}
```

---

## 3. LLM được bind với tools ra sao?

LLM được bind với tools bằng:

```python
llm_with_tools = llm.bind_tools(TOOLS)
```

Trong đó:

```python
TOOLS = [search_legal_database, calculate_damages]
```

Sau khi bind:

* LLM biết các tool tồn tại,
* biết schema arguments,
* có thể tự quyết định gọi tool nào.

---

# Bài Tập 2.1 — Thêm Knowledge Base Entry

Đã thêm entry mới vào `LEGAL_KNOWLEDGE`:

```python
{
    "id": "labor_law",
    "keywords": ["lao động", "sa thải", "hợp đồng lao động", "labor", "termination"],
    "text": (
        "Theo Bộ luật Lao động Việt Nam 2019, người sử dụng lao động có thể "
        "đơn phương chấm dứt hợp đồng trong các trường hợp: (1) người lao động "
        "thường xuyên không hoàn thành công việc; (2) bị ốm đau, tai nạn đã điều trị "
        "12 tháng chưa khỏi; (3) thiên tai, hỏa hoạn; (4) người lao động đủ tuổi nghỉ hưu."
    ),
}
```

Entry này giúp legal database có thêm dữ liệu về:

* luật lao động,
* chấm dứt hợp đồng lao động,
* sa thải,
* termination.

---

# Bài Tập 2.2 — Tạo Tool Mới

Đã tạo tool mới:

```python
@tool
def check_statute_of_limitations(case_type: str) -> str:
    """Kiểm tra thời hiệu khởi kiện theo loại vụ án.
    
    Args:
        case_type: Loại vụ án (contract, tort, property)
    """
    limits = {
        "contract": "4 năm (UCC § 2-725)",
        "tort": "2-3 năm tùy bang",
        "property": "5 năm",
    }

    return limits.get(case_type.lower(), "Không xác định")
```

---

## Thêm tool vào danh sách tools

```python
TOOLS = [
    search_legal_database,
    calculate_damages,
    check_statute_of_limitations,
]
```

---

## Kết quả

```python
======================================================================
STAGE 2: LLM + RAG / Tools
======================================================================

[How it works]
  1. LLM receives tools (search_legal_database, calculate_damages)
  2. LLM decides which tools to call and with what arguments
  3. We execute the tools and feed results back to the LLM
  4. LLM generates a final answer grounded in retrieved data

Question: Thời hạn khởi kiện tranh chấp hợp đồng là khi nào?
----------------------------------------------------------------------

>>> Step 1: Asking LLM (with tools bound)...

>>> Step 2: LLM requested 2 tool call(s):

  Tool: check_statute_of_limitations
  Args: {'case_type': 'contract'}
  Result: 4 năm (UCC § 2-725)

  Tool: search_legal_database
  Args: {'query': 'thời hiệu khởi kiện tranh chấp hợp đồng'}
  Result: No relevant legal sources found for this query.

>>> Step 3: LLM generating final answer with tool results...

Dựa trên kết quả tra cứu, tôi xin cung cấp thông tin về thời hiệu khởi kiện tranh chấp hợp đồng:

## Thời Hiệu Khởi Kiện Tranh Chấp Hợp Đồng

**Thời hạn chung: 4 năm**

Theo quy định tại UCC § 2-725 (Uniform Commercial Code - Bộ luật Thương mại Thống nhất của Hoa Kỳ), thời hiệu khởi kiện đối với tranh chấp hợp đồng là **4 năm**.

### Điểm Quan Trọng Cần Lưu Ý:

1. **Thời điểm tính**: Thời hiệu thường được tính từ khi quyền khởi kiện phát sinh, tức là:
   - Khi hợp đồng bị vi phạm
   - Khi bên bị thiệt hại biết hoặc phải biết về việc vi phạm
   - Khi thiệt hại xảy ra

2. **Không thể rút ngắn**: Các bên không được thỏa thuận rút ngắn thời hiệu xuống dưới 1 năm

3. **Có thể gia hạn**: Trong một số trường hợp, thời hiệu có thể được gia hạn lên đến 5 năm nếu có thỏa thuận

### Lưu Ý Đặc Biệt:

- Đối với **hợp đồng bằng văn bản** có thể có thời hiệu khác nhau tùy theo quy định của từng bang
- **Hợp đồng miệng** thường có thời hiệu ngắn hơn
- Một số loại hợp đồng đặc biệt (bất động sản, bảo hiểm, lao động) có thể có quy định riêng

### Khuyến Nghị:

Nếu bạn đang trong tình huống tranh chấp hợp đồng cụ thể, tôi khuyên bạn nên:
- Xác định rõ thời điểm vi phạm hợp đồng xảy ra
- Tham khảo luật sư để đảm bảo không bỏ lỡ thời hiệu
- Chuẩn bị hồ sơ, chứng cứ sớm

Bạn có cần tôi tìm hiểu thêm về loại hợp đồng cụ thể nào không?

----------------------------------------------------------------------
[Improvements over Stage 1]
  + Grounded: answers cite specific statutes (DTSA, UCC, etc.)
  + Tool use: can search databases and calculate damages
  + More accurate: retrieval reduces hallucination risk

[Limitations of Stage 2]
  - Manual orchestration: we wrote the tool-call loop ourselves
  - Single pass: only one round of tool calls
  - No reasoning loop: LLM can't decide to search again if needed

Next: Stage 3 wraps this in an autonomous ReAct agent loop.
======================================================================
```

Sau khi thêm tool:

* LLM có thể tự động kiểm tra thời hiệu khởi kiện,
* hỗ trợ trả lời các câu hỏi pháp lý liên quan đến statute of limitations,
* tăng khả năng reasoning và grounding của hệ thống.

## Phần 3: Single Agent với ReAct (25 phút)

## 1. `create_react_agent()` — Magic Function

Trong file `stages/stage_3_single_agent/main.py`:

```python
from langgraph.prebuilt import create_react_agent
```

và:

```python
graph = create_react_agent(
    model=llm,
    tools=TOOLS,
    prompt=SYSTEM_PROMPT
)
```

Đây là “magic function” vì:

* tự tạo ReAct agent,
* tự quản lý reasoning loop,
* tự gọi tools,
* tự quan sát kết quả,
* tự quyết định bước tiếp theo.

Agent hoạt động theo mô hình:

```text
Thought → Action → Observation → Thought → ...
```

---

## 2. So sánh với Stage 2

### Stage 2

Programmer phải tự viết:

* tool loop,
* execute tool,
* append ToolMessage,
* gọi LLM lại.

Ví dụ:

```python
response = await llm_with_tools.ainvoke(messages)

for tc in response.tool_calls:
    result = await tool_fn.ainvoke(tc["args"])
    messages.append(ToolMessage(...))

final_response = await llm_with_tools.ainvoke(messages)
```

Đây là manual orchestration.

---

### Stage 3

Chỉ cần:

```python
graph = create_react_agent(...)
```

LangGraph tự:

* reasoning,
* tool calling,
* observation,
* looping.

Không cần manual tool loop nữa.

---

## 3. `agent_executor.invoke()` / `graph.astream()`

Trong Stage 3:

```python
async for chunk in graph.astream(inputs, stream_mode="updates"):
```

Chỉ cần gọi:

* một lần,
* agent tự chạy toàn bộ workflow.

Khác với Stage 2:

* phải gọi LLM nhiều lần thủ công.

---

# Bài Tập 3.1 — Thêm Tool Tra Cứu Án Lệ

Đã thêm tool mới:

```python
@tool
def search_case_law(keywords: str) -> str:
    """Tìm kiếm án lệ theo từ khóa.
    
    Args:
        keywords: Từ khóa tìm kiếm
    """
    cases = {
        "breach": "Hadley v. Baxendale (1854) - Consequential damages",
        "negligence": "Donoghue v. Stevenson (1932) - Duty of care",
        "contract": "Carlill v. Carbolic Smoke Ball Co (1893) - Unilateral contract",
    }

    for key, case in cases.items():
        if key in keywords.lower():
            return case

    return "Không tìm thấy án lệ phù hợp"
```

---

## Thêm vào tools list

```python
TOOLS = [
    search_legal_database,
    calculate_penalty,
    check_compliance_requirements,
    search_case_law,
]
```

---

## Test với câu hỏi breach of contract

Ví dụ:

```python
QUESTION = "Những án lệ nào áp dụng cho các tranh chấp vi phạm hợp đồng?"
```

---

## Kết quả 

```python
======================================================================
STAGE 3: Single Agent (ReAct Loop)
======================================================================

[How it works]
  1. An autonomous agent receives a complex multi-part question
  2. It reasons about what tools to call (Think)
  3. It calls a tool (Act)
  4. It observes the result and decides next steps (Observe)
  5. It repeats until it has enough information for a final answer

Question: Những án lệ nào áp dụng cho các tranh chấp vi phạm hợp đồng?
----------------------------------------------------------------------
D:\AI_inAction_2026\Batch02-Day9_Multi-Agent_MCP-A2A-main\stages\stage_3_single_agent\main.py:225: LangGraphDeprecatedSinceV10: create_react_agent has been moved to `langchain.agents`. Please update your import to `from langchain.agents import create_agent`. Deprecated in LangGraph V1.0 to be removed in V2.0.
  graph = create_react_agent(model=llm, tools=TOOLS, prompt=SYSTEM_PROMPT)

[Step 1] THINK + ACT (node: agent)
  Tool: search_case_law
  Args: {'keywords': 'vi phạm hợp đồng tranh chấp'}
  Tool: search_legal_database
  Args: {'query': 'án lệ tranh chấp vi phạm hợp đồng'}

[Step 2] OBSERVE (node: tools)
  Result: No relevant legal sources found.

[Step 3] OBSERVE (node: tools)
  Result: Không tìm thấy án lệ phù hợp

[Step 4] FINAL ANSWER (node: agent)
----------------------------------------------------------------------
Rất tiếc, hệ thống hiện tại không tìm thấy các án lệ cụ thể trong cơ sở dữ liệu pháp lý về tranh chấp vi phạm hợp đồng. Tuy nhiên, tôi có thể cung cấp cho bạn thông tin tổng quan về vấn đề này:

## Các Án Lệ Về Vi Phạm Hợp Đồng

**Khung pháp lý chung:**
- Các tranh chấp vi phạm hợp đồng ở Việt Nam chủ yếu được điều chỉnh bởi Bộ luật Dân sự 2015
- Án lệ được áp dụng theo Nghị quyết 03/2015/NQ-HĐTP về công bố án lệ của Hội đồng Thẩm phán Tòa án nhân dân tối cao

**Các loại vi phạm hợp đồng thường gặp:**
1. **Vi phạm nghĩa vụ thanh toán** - không thanh toán đúng hạn hoặc đủ số tiền
2. **Vi phạm nghĩa vụ giao hàng** - không giao hàng đúng thời gian, số lượng, chất lượng
3. **Vi phạm nghĩa vụ bảo mật** - tiết lộ thông tin mật
4. **Đơn phương chấm dứt hợp đồng trái pháp luật**

**Biện pháp khắc phục:**
- Bồi thường thiệt hại
- Buộc thực hiện đúng hợp đồng
- Phạt vi phạm (nếu có thỏa thuận)
- Đơn phương chấ

----------------------------------------------------------------------
[Improvements over Stage 2]
  + Autonomous: agent decides which tools to call and when
  + Multi-step reasoning: can search, calculate, search again
  + Handles complex queries: breaks problems into sub-tasks

[Limitations of Stage 3]
  - Single agent: one LLM handles all domains (law, tax, compliance)
  - No specialisation: same system prompt for all legal areas
  - Bottleneck: sequential tool calls, no parallelism

Next: Stage 4 splits this into specialised agents that work in parallel.
======================================================================
```

---

# Bài Tập 3.2 — Debug Agent Reasoning

Đã thêm:

```python
verbose=True
```

vào:

```python
graph = create_react_agent(
    model=llm,
    tools=TOOLS,
    prompt=SYSTEM_PROMPT,
    verbose=True,
)
```

---

## Ý nghĩa của `verbose=True`

Cho phép:

* xem reasoning process,
* xem từng tool call,
* debug agent behavior,
* hiểu agent đang “suy nghĩ” gì.

---

## Khi chạy sẽ thấy

```python
======================================================================
STAGE 2: LLM + RAG / Tools
======================================================================

[How it works]
  1. LLM receives tools (search_legal_database, calculate_damages)
  2. LLM decides which tools to call and with what arguments
  3. We execute the tools and feed results back to the LLM
  4. LLM generates a final answer grounded in retrieved data

Question: Thời hạn khởi kiện tranh chấp hợp đồng là khi nào?
----------------------------------------------------------------------

>>> Step 1: Asking LLM (with tools bound)...

>>> Step 2: LLM requested 2 tool call(s):

  Tool: check_statute_of_limitations
  Args: {'case_type': 'contract'}
  Result: 4 năm (UCC § 2-725)

  Tool: search_legal_database
  Args: {'query': 'thời hiệu khởi kiện tranh chấp hợp đồng'}
  Result: No relevant legal sources found for this query.

>>> Step 3: LLM generating final answer with tool results...

Dựa trên kết quả tra cứu, tôi xin cung cấp thông tin về thời hiệu khởi kiện tranh chấp hợp đồng:

## Thời Hiệu Khởi Kiện Tranh Chấp Hợp Đồng

**Thời hạn chung: 4 năm**

Theo quy định tại UCC § 2-725 (Uniform Commercial Code - Bộ luật Thương mại Thống nhất), thời hiệu khởi kiện đối với tranh chấp hợp đồng là **4 năm**.

### Điểm Quan Trọng Cần Lưu Ý:

1. **Thời điểm tính**: Thời hiệu thường được tính từ khi quyền khởi kiện phát sinh, tức là:
   - Khi hợp đồng bị vi phạm
   

----------------------------------------------------------------------
[Improvements over Stage 1]
  + Grounded: answers cite specific statutes (DTSA, UCC, etc.)
  + Tool use: can search databases and calculate damages
  + More accurate: retrieval reduces hallucination risk

[Limitations of Stage 2]
  - Manual orchestration: we wrote the tool-call loop ourselves
  - Single pass: only one round of tool calls
  - No reasoning loop: LLM can't decide to search again if needed

Next: Stage 3 wraps this in an autonomous ReAct agent loop.
======================================================================
```

---

# Kết luận

Stage 3 là bước chuyển từ:

* LLM + tools
  → sang
* Autonomous AI Agent.

Các điểm nổi bật:

* ReAct reasoning loop,
* autonomous tool calling,
* multi-step reasoning,
* dynamic planning,
* grounded responses.

## Phần 4: Multi-Agent In-Process (30 phút)

**Bước 2:** Phân tích kiến trúc

Mở `stages/stage_4_milti_agent/main.py`:

1. Tìm `class State(TypedDict)` — đây là shared state

```python
class LegalState(TypedDict):
    question: str
    law_analysis: str
    needs_tax: bool
    needs_compliance: bool
    tax_result: Annotated[str, _last_wins]
    compliance_result: Annotated[str, _last_wins]
    final_answer: str
```

2. Tìm các agent functions: `law_agent`, `tax_agent`, `compliance_agent`

| Vai trò agent    | Function hiện tại            |
| ---------------- | ---------------------------- |
| Law Agent        | `analyze_law`                |
| Tax Agent        | `call_tax_specialist`        |
| Compliance Agent | `call_compliance_specialist` |

3. Tìm `Send()` API — dispatch parallel tasks

Import Send()
```python
from langgraph.constants import Send
```

Function dùng Send() thật sự
```python
def route_to_specialists(state: LegalState) -> list[Send]:
    #Đoạn quan trọng nhất:
    if state.get("needs_tax"):
        sends.append(Send("call_tax_specialist", state))
    if state.get("needs_compliance"):
        sends.append(Send("call_compliance_specialist", state))
```
Ở đây:
```python
Send("call_tax_specialist", state)
```
nghĩa là gửi shared state sang node call_tax_specialist


4. Xem `graph.add_node()` và `graph.add_edge()`

**Bước 3:** Vẽ graph

```python
# Thêm vào cuối file main.py
from IPython.display import Image, display
display(Image(graph.get_graph().draw_mermaid_png()))
```

**Bài Tập 4.1:** Thêm agent mới

Tạo `privacy_agent` chuyên về GDPR và privacy law:

```python
def privacy_agent(state: State) -> dict:
    """Agent chuyên về luật bảo vệ dữ liệu cá nhân."""
    llm = get_llm()
  
    prompt = f"""Bạn là chuyên gia về GDPR và luật bảo vệ dữ liệu cá nhân.
  
Câu hỏi gốc: {state['question']}
Phân tích pháp lý: {state.get('law_analysis', 'N/A')}

Hãy phân tích các vấn đề về privacy và GDPR (nếu có).
"""
  
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"privacy_analysis": response.content}
```

Thêm node này vào graph và kết nối với `aggregate_results`.

> **Đã hoàn thành:** Đã thêm specialist privacy vào [stages/stage_4_milti_agent/main.py](stages/stage_4_milti_agent/main.py) (đặt tên `call_privacy_specialist` cho nhất quán với các node khác trong file):
>
> - Thêm field `needs_privacy: bool` và `privacy_result: Annotated[str, _last_wins]` vào `LegalState`.
> - Thêm hàm `call_privacy_specialist(state)` dùng prompt GDPR/CCPA như đề bài.
> - Đăng ký node + edge `call_privacy_specialist → aggregate`, và `aggregate` đã gộp thêm mục `## Data Privacy Analysis`.

**Bài Tập 4.2:** Implement conditional routing

Sửa `check_routing` để chỉ gọi privacy_agent khi câu hỏi có từ khóa "data", "privacy", "gdpr":

```python
def check_routing(state: State) -> list[Send]:
    question_lower = state["question"].lower()
    tasks = []
  
    if any(kw in question_lower for kw in ["tax", "irs", "thuế"]):
        tasks.append(Send("tax_agent", state))
  
    if any(kw in question_lower for kw in ["compliance", "sec", "regulation"]):
        tasks.append(Send("compliance_agent", state))
  
    if any(kw in question_lower for kw in ["data", "privacy", "gdpr", "dữ liệu"]):
        tasks.append(Send("privacy_agent", state))
  
    return tasks if tasks else [Send("aggregate_results", state)]
```

> **Đã hoàn thành:** Routing có điều kiện đã được hiện thực trong [stages/stage_4_milti_agent/main.py](stages/stage_4_milti_agent/main.py). Vì file tách `check_routing` (đặt cờ) và `route_to_specialists` (trả về `list[Send]`), logic keyword được đặt trong `check_routing`:
> `needs_privacy = any(kw in question_lower for kw in ["data", "privacy", "gdpr", "dữ liệu"])`,
> còn `route_to_specialists` chỉ `Send("call_privacy_specialist", state)` khi `needs_privacy` là `True`. Nhờ vậy privacy specialist chỉ chạy khi câu hỏi thực sự liên quan dữ liệu/GDPR, tiết kiệm chi phí gọi LLM.

---

## Phần 5: Distributed A2A System (15 phút)

### Lý Thuyết

**A2A (Agent-to-Agent) Protocol:** Chuẩn giao tiếp giữa các agents qua HTTP.

**Khác biệt với Stage 4:**

- Mỗi agent là một service độc lập
- Giao tiếp qua HTTP thay vì in-process
- Dynamic discovery qua Registry
- Có thể scale từng agent riêng biệt

**Kiến trúc:**

```
Registry (10000) ← agents register on startup
    ↓
Customer Agent (10100) → Law Agent (10101)
                              ↓
                    ┌─────────┴─────────┐
                    ↓                   ↓
            Tax Agent (10102)   Compliance Agent (10103)
```

### Thực Hành

**Bước 1:** Khởi động toàn bộ hệ thống

```bash
./start_all.sh
```

Chờ ~10 giây để tất cả services khởi động.

**Bước 2:** Test hệ thống

```bash
uv run python test_client.py
```

**Bước 3:** Quan sát logs

Mở 5 terminal tabs và xem logs của từng service:

- Registry: port 10000
- Customer Agent: port 10100
- Law Agent: port 10101
- Tax Agent: port 10102
- Compliance Agent: port 10103

**Bài Tập 5.1:** Trace request flow

Trong logs, tìm `trace_id` và theo dõi request đi qua các agents. Vẽ sequence diagram.

> **Trả lời:** Mỗi request được gắn một `trace_id` (sinh ở Customer Agent) và được truyền nguyên vẹn qua metadata của message A2A (`metadata={"trace_id": ...}` trong `common/a2a_client.py`). Grep cùng `trace_id` trên cả 5 service sẽ thấy đường đi: Customer → Law → (song song) Tax + Compliance → trả ngược về. Sequence diagram:
>
> ```mermaid
> sequenceDiagram
>     participant C as test_client
>     participant CU as Customer (10100)
>     participant R as Registry (10000)
>     participant L as Law (10101)
>     participant T as Tax (10102)
>     participant CO as Compliance (10103)
>     C->>CU: send_message(question)
>     CU->>R: discover("law_question")
>     R-->>CU: endpoint Law
>     CU->>L: delegate(question, trace_id, depth=1)
>     L->>L: analyze_law + check_routing
>     par parallel (Send API)
>         L->>R: discover("tax_question")
>         L->>T: delegate(depth=2)
>         T-->>L: tax_result
>     and
>         L->>R: discover("compliance_question")
>         L->>CO: delegate(depth=2)
>         CO-->>L: compliance_result
>     end
>     L->>L: aggregate
>     L-->>CU: final_answer
>     CU-->>C: response
> ```

**Bài Tập 5.2:** Test dynamic discovery

1. Dừng Tax Agent (Ctrl+C)
2. Chạy lại `test_client.py`
3. Quan sát lỗi và cách hệ thống xử lý

> **Trả lời:** Khi Tax Agent dừng, có hai khả năng tùy thời điểm:
>
> - **Nếu Tax Agent đã unregister/registry không trả endpoint:** `discover("tax_question")` thất bại → nhánh `call_tax` ném exception và được bắt trong `try/except`, trả về `"[Tax analysis unavailable: ...]"`. Hệ thống **không sập** — Compliance vẫn chạy và `aggregate` vẫn tổng hợp được câu trả lời (chỉ thiếu phần thuế).
> - **Nếu registry vẫn còn endpoint cũ (stale):** việc gọi HTTP tới Tax Agent sẽ lỗi connection refused; cũng bị `except` bắt và trả về cùng thông báo unavailable.
>
> Đây chính là tính **fault tolerance**: lỗi của một agent được cô lập (graceful degradation), không kéo đổ toàn hệ thống. Dynamic discovery cho phép registry phản ánh agent nào đang sống; nếu Tax Agent khởi động lại và register lại, lần chạy sau nó lại được dùng bình thường mà không cần sửa code.

**Bài Tập 5.3:** Modify agent behavior

Sửa `tax_agent/graph.py`, thay đổi system prompt để agent trả lời ngắn gọn hơn. Restart tax agent và test lại.

> **Trả lời / Hướng dẫn:** Mở `tax_agent/graph.py`, tìm system prompt của tax specialist và thêm ràng buộc độ dài, ví dụ thêm câu *"Trả lời thật ngắn gọn, tối đa 3 gạch đầu dòng, dưới 80 từ."* vào cuối prompt. Sau đó **restart riêng** Tax Agent (Ctrl+C tiến trình port 10102 rồi chạy lại `uv run python -m tax_agent`) và chạy lại `test_client.py`. Vì kiến trúc A2A loose-coupling, chỉ cần restart đúng service đó — Registry, Law, Compliance, Customer vẫn giữ nguyên; đây là ưu điểm về khả năng triển khai/scale độc lập so với monolith ở Stage 4.

---

## Phần 6: Tổng Kết & Mở Rộng (10 phút)

### So Sánh 5 Stages

| Stage | Pattern         | Use Case                                 | Complexity |
| ----- | --------------- | ---------------------------------------- | ---------- |
| 1     | Direct LLM      | Câu hỏi đơn giản, không cần tools | ⭐         |
| 2     | LLM + Tools     | Cần tra cứu data hoặc tính toán     | ⭐⭐       |
| 3     | ReAct Agent     | Tự động orchestration, multi-step     | ⭐⭐⭐     |
| 4     | Multi-Agent     | Nhiều domains, parallel processing      | ⭐⭐⭐⭐   |
| 5     | Distributed A2A | Production, scalable, fault-tolerant     | ⭐⭐⭐⭐⭐ |

### Câu Hỏi Ôn Tập

1. Khi nào nên dùng single agent thay vì multi-agent?
2. Ưu điểm của A2A protocol so với gRPC hoặc REST thông thường?
3. Làm thế nào để prevent infinite delegation loops trong A2A?
4. Tại sao cần Registry service? Có thể hardcode URLs không?

> **Trả lời:**
>
> 1. **Single vs multi-agent:** Dùng **single agent** khi bài toán thuộc một domain, số tool ít, độ phức tạp thấp/trung bình — đơn giản hơn, ít overhead, dễ debug, rẻ hơn (ít lần gọi LLM). Chuyển sang **multi-agent** khi cần nhiều chuyên môn khác nhau (luật + thuế + compliance), muốn chạy song song để giảm latency, hoặc cần scale/maintain từng phần độc lập. Nguyên tắc: bắt đầu đơn giản, chỉ tách agent khi prompt/tool của một agent trở nên quá tải hoặc khi cần parallel.
> 2. **A2A so với gRPC/REST thuần:** A2A xây trên HTTP nhưng **chuẩn hóa riêng cho agent**: có Agent Card (`/.well-known/agent.json`) để mô tả khả năng (skills) → cho phép **dynamic discovery**; có khái niệm `task`, `context_id`, message/parts và streaming phù hợp hội thoại nhiều bước; truyền metadata như `trace_id`, `delegation_depth` xuyên suốt. gRPC/REST chỉ là transport, bạn phải tự định nghĩa contract, discovery, định danh task... A2A đem lại tính **liên thông (interoperability)** giữa agent của các bên khác nhau mà không cần thỏa thuận API riêng.
> 3. **Chống infinite delegation loop:** Truyền `delegation_depth` trong metadata và **tăng dần mỗi lần delegate**; mỗi agent kiểm tra ngưỡng `MAX_DELEGATION_DEPTH` (trong repo = 3) — khi đạt ngưỡng thì `check_routing` trả `needs_tax=False, needs_compliance=False`, dừng việc gọi tiếp. Kết hợp với `trace_id`/`context_id` để phát hiện vòng lặp, và timeout cho mỗi HTTP call. Đây là cơ chế then chốt tránh A→B→A→B vô hạn.
> 4. **Tại sao cần Registry:** Registry cho phép **service discovery động** — agent tự đăng ký khi khởi động và client/agent khác tra cứu theo *khả năng* ("tax_question") thay vì địa chỉ cứng. Có thể hardcode URL được (đơn giản hơn cho demo), nhưng sẽ mất tính linh hoạt: không tự phát hiện agent chết/mới, khó scale nhiều instance (load-balancing), khó thay đổi cấu hình mà không sửa code và redeploy. Registry là một dạng *indirection* giúp hệ thống loose-coupled và mở rộng được trong môi trường production.

### Bài Tập Nâng Cao (Tự Học)

**Challenge 1:** Thêm memory/conversation history

Implement conversation memory để agent nhớ các câu hỏi trước đó.

**Challenge 2:** Add authentication

Thêm API key authentication cho các A2A endpoints.

**Challenge 3:** Implement retry logic

Khi một agent fail, tự động retry với exponential backoff.

**Challenge 4:** Monitoring & Observability

Tích hợp LangSmith hoặc Prometheus để monitor agent performance.

---

## Tài Liệu Tham Khảo

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [A2A Protocol Spec](https://github.com/google/A2A)
- [OpenRouter API](https://openrouter.ai/docs)
- Architecture diagrams: `docs/*.svg`

## Hỗ Trợ

Nếu gặp vấn đề:

1. Check `.env` file có đúng API key không
2. Đảm bảo tất cả ports (10000-10103) không bị chiếm
3. Xem logs trong terminal để debug
4. Đọc error messages cẩn thận — thường có hint rõ ràng

---

## **Bài Tập Cộng Điểm:**

Sau khi chạy full Stage 5 (test_client.py) trả lời 2 câu hỏi:

- Latency (Tổng thời gian trả lời 1 câu hỏi của hệ thống) là bao nhiêu giây?
- Đề xuất phương án giảm latency và demo + show thời gian xử lý đã giảm được khi apply phương án?

> **Trả lời:**
>
> **1. Đo latency:** `test_client.py` đã được bổ sung đo thời gian end-to-end (dùng `time.perf_counter()` bao quanh `client.send_message`) và in dòng `[Latency] Total end-to-end response time: X.XX seconds`. **Kết quả đo thực tế với model `google/gemma-4-31b-it:free`: `153.18 giây`** cho câu hỏi mẫu (luật + thuế + compliance). Latency cao như vậy chủ yếu do: (a) chuỗi gọi LLM tuần tự Customer → Law (`analyze_law` + `check_routing` + `aggregate`) cộng dồn; (b) Tax + Compliance tuy chạy *song song* nhưng mỗi nhánh là một ReAct agent nhiều bước; (c) model **free bị rate-limit (HTTP 429 Too Many Requests)** nên client tự retry, cộng thêm thời gian chờ. Nếu dùng model trả phí/nhanh hơn, latency sẽ giảm mạnh.
>
> **Phân tích nguồn latency:** Phần lớn thời gian là **độ trễ inference của LLM**, không phải HTTP overhead. Các node *tuần tự* (analyze_law → check_routing → ... → aggregate) cộng dồn; các specialist đã chạy *song song* nên không cộng dồn.
>
> **2. Đề xuất phương án giảm latency (kèm cách demo so sánh):**
>
> | Phương án                                   | Cách làm                                                                                                                        | Kỳ vọng                                  |
> | ---------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------ |
> | **Bỏ bớt node LLM thừa**              | Gộp `check_routing` bằng keyword-matching (không gọi LLM) thay vì hỏi LLM để quyết định routing                      | Tiết kiệm trọn 1 lần gọi LLM (~3–8s) |
> | **Dùng model nhỏ/nhanh cho node phụ** | Đặt `OPENROUTER_MODEL` của routing/specialist sang model nhẹ (vd `*-haiku`/`*-mini`), giữ model lớn cho `aggregate` | Giảm 30–50% thời gian mỗi node phụ    |
> | **Streaming output**                     | Stream token của `aggregate` về client thay vì chờ trọn câu trả lời                                                     | Giảm*perceived latency* rõ rệt        |
> | **Giảm token**                          | Rút gọn system prompt + giới hạn `max_tokens` cho specialist                                                                | Giảm thời gian sinh token                |
> | **Đảm bảo parallel thực sự**        | Kiểm tra Tax + Compliance chạy đồng thời qua `Send` (đã có)                                                             | Tránh cộng dồn tuần tự                |
>
> **Demo so sánh (đã triển khai):** Phương án **"bỏ node LLM thừa"** đã được code thật trong [law_agent/graph.py](law_agent/graph.py). Hàm `check_routing` giờ đọc biến môi trường `ROUTING_MODE`:
>
> - `ROUTING_MODE=llm` (baseline): gọi thêm **1 lần LLM** để hỏi nên route tới Tax/Compliance hay không.
> - `ROUTING_MODE=keyword` (tối ưu): quyết định routing bằng **keyword-matching thuần Python**, *không gọi LLM* → loại bỏ hẳn 1 round-trip LLM trên đường tuần tự.
>
> Cách chạy demo:
>
> ```powershell
> # Baseline
> $env:ROUTING_MODE="llm"; uv run python -m law_agent   # (cùng registry + tax + compliance + customer)
> uv run python test_client.py                          # đọc dòng [Latency]
>
> # Tối ưu
> $env:ROUTING_MODE="keyword"; uv run python -m law_agent
> uv run python test_client.py                          # đọc dòng [Latency] mới
> ```
> **Kết quả đo & ghi chú trung thực:**
>
> | Lần chạy             | Chế độ routing | Latency đo được | Trạng thái                                                |
> | ---------------------- | ----------------- | ------------------- | ----------------------------------------------------------- |
> | Run sạch ban đầu    | `llm`           | **153.18 s**  | ✅ Thành công (đầy đủ tax + compliance)               |
> | Re-test (cùng phiên) | `llm`           | 95.05 s             | ⚠️ Kết thúc bằng HTTP 429 (đã chạm hạn mức ngày) |
> | Re-test (cùng phiên) | `keyword`       | 4.57 s              | ⚠️ Fail-fast tại LLM call đầu của Customer (HTTP 429) |
>
> **Vì sao chưa có phép đo "thành công" sạch cho `keyword`:** API key model free `google/gemma-4-31b-it:free` có hạn mức **50 requests/ngày**, và trong lúc đo lại đã **cạn quota** (`X-RateLimit-Remaining: 0`, lỗi `429 Rate limit exceeded: free-models-per-day`). Khi quota đã hết, mọi request đều fail nên hai con số 95.05s/4.57s **không phản ánh hiệu năng thật** — chúng chỉ cho thấy mỗi đường đi chạm "bức tường rate-limit" nhanh hay chậm.
>
> **Bằng chứng phân tích từ log run sạch (153.18s):** Riêng node `check_routing` ở chế độ `llm` tốn **~27 giây** cho 1 lần gọi LLM (đo từ timestamp giữa `analyze_law` xong → `check_routing` xong trong log Law Agent). Chế độ `keyword` loại bỏ hoàn toàn lần gọi này, nên **kỳ vọng giảm ~27s (~18%)** trên đường tuần tự, đồng thời **giảm 1 request lên model free → giảm rủi ro 429**. Để có bảng *Before → After* sạch, chạy lại đúng 2 lệnh trên **sau khi quota reset** (hoặc nạp 10 credits để mở 1000 req/ngày).