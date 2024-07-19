模块梳理
- Presentation: 负责ppt元素的数据存储、重建、表示(html等)
- LayoutPlaner: 负责提供BoxModel的反馈，计算元素的真实大小，根据生成结果微调ppt元素
- ImageLabeler: 对ppt原有以及论文中的幻灯片进行caption，以及对picture元素进行bg判断
- TemplateInduction：根据解析出的内容，判断哪些文本元素属于content哪些属于bg，对于textframe给出对应的标签
- llms: 负责实例化各种用到的大模型，以及交互操作
- utils: 工具类