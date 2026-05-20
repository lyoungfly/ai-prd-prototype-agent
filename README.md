# ai-prd-prototype-agent
我构建了一个“AI 产品需求到原型自动生成 Agent”，解决早期产品设计中需求不清、PRD 编写慢、原型反复修改、研发理解偏差大的问题。核心流程是：用户输入一句产品想法或业务痛点后，需求分析 Agent 会先追问关键场景、目标用户和约束条件；随后 PRD Agent 自动生成用户故事、功能清单、流程图和验收标准；竞品分析 Agent 检索公开资料并总结差异化方案；原型 Agent 根据 PRD 生成页面结构、交互说明和可导入 Figma/前端的组件描述；最后评审 Agent 从商业价值、技术可行性、用户体验和风险角度打分，并给出修改建议。
运行方式
进入项目目录：

bash
cd ai-prd-prototype-agent
安装依赖：

bash
pip install -r requirements.txt
启动服务：

bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
浏览器打开：

bash
http://127.0.0.1:8000
