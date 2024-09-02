# 模块梳理

## Presentation
负责ppt元素的数据存储、重建、表示(html等)。

文件: `presentation.py`

## Multimodal
对ppt原有以及论文中的幻灯片进行caption，以及对picture元素进行bg判断。

文件: `multimodal.py`

## TemplateInduct
根据解析出的内容，判断哪些文本元素属于content哪些属于bg，对于textframe给出对应的标签。

文件: `template_induct.py`

## llms
负责实例化各种用到的大模型，以及交互操作。

文件: `llms.py`

## utils
工具类。

文件: `utils.py`

## PPTAgent
负责生成PPT的主要逻辑，包括文档解析、图片标注、模板生成等。

文件: `agent.py`

## apis
提供设置图片、选择模板、克隆形状、删除形状等API。

文件: `apis.py`

## main
主程序入口，负责调用各个模块生成PPT。

文件: `main.py`
