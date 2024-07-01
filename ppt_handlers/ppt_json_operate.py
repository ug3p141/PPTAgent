import os
import json
from ppt_handlers.utils import Config

base_config = Config()


def get_skeleton_from_json(ppt_json, ret_all=False):
    """
    从json解析结果中，只分离出数据，去掉样式
    :param ppt_json:
    :return: 适合element-plus的cascader的数据格式
    """
    type_en2zh = {'text': '文本', 'image': '图片', 'table': '表格', 'chart': '图形'}
    ppt_info = {thek: thev for thek, thev in ppt_json.items() if thek != 'slide_pages'}
    json_main_data, leaf_id_list, id2content = [], [], {}  # leaf_id_list是所有叶节点的id列表
    for page_info in ppt_json.get('slide_pages', []):
        main_page_info = {'id': page_info['id'], 'value': page_info['id'], 'label': f"第{int(page_info['id']) + 1}页"}
        shape_list = []
        for shape_info in page_info.get('slide_data', []):
            shape_type = shape_info['type']
            shape_type_zh = type_en2zh.get(shape_type, shape_type)
            main_shape_info = {'id': shape_info['id'], 'type': shape_type, 'value': shape_info['id'],
                               'label': '形状' + shape_info['id']}
            if shape_type == 'text':
                para_list = []
                for para in shape_info.get('frame_data', []):
                    if para.get('data'):
                        if para['data'][0].strip():
                            para_list.append({'id': para['id'], 'label': shape_type_zh + para['id'],
                                              'value': para['id'], 'data': para['data']})
                        if para['data'][0].strip():
                            leaf_id_list.append(para['id'])
                        if ret_all:
                            id2content[para['id']] = para
                        else:
                            id2content[para['id']] = para['data']
                if para_list:
                    main_shape_info['children'] = para_list
            elif shape_info.get('frame_data'):
                main_shape_info['data'] = shape_info['frame_data']
                leaf_id_list.append(shape_info['id'])
                if ret_all:
                    id2content[shape_info['id']] = shape_info
                else:
                    id2content[shape_info['id']] = shape_info['frame_data']
            if main_shape_info.get('data') or main_shape_info.get('children'):
                shape_list.append(main_shape_info)
        if shape_list:
            main_page_info['children'] = shape_list
        json_main_data.append(main_page_info)
    return ppt_info, json_main_data, leaf_id_list, id2content


def get_elements_by_ids(id_list, json_fname, ret_skeleton=True):
    json_data = json.load(open(os.path.join(base_config.PPT_DIR, json_fname), encoding='utf-8'))
    slide_pages = json_data.get('slide_pages', [])
    id_content_list = []
    for the_id in id_list:
        parts = the_id.split('-')
        try:
            found_it = False
            page_idx = int(parts[0])
            for a_shape in slide_pages[page_idx]['slide_data']:  # shape、para实际的idx和id中的不一样，因为解析结束时按位置排序了
                if a_shape['id'] == the_id:
                    if not ret_skeleton:
                        id_content_list.append(a_shape)
                    else:
                        content = a_shape['frame_data'][0] if a_shape['type'] == 'picture' else a_shape['frame_data']
                        id_content_list.append({'id': the_id, 'content': content, 'type': a_shape['type']})
                    break
                elif a_shape['type'] == 'text':
                    for a_para in a_shape.get('frame_data', []):
                        if a_para['id'] == the_id:
                            if not ret_skeleton:
                                id_content_list.append(a_para)
                            else:
                                id_content_list.append(
                                    {'id': the_id, 'content': a_para['data'][0], 'type': a_shape['type']})
                            found_it = True
                            break
                if found_it:
                    break
        except:
            print('id不存在', the_id)
    return id_content_list


def delete_elements_by_ids(id_list, json_fname):
    json_data = json.load(open(os.path.join(base_config.PPT_DIR, json_fname), encoding='utf-8'))
    slide_pages = json_data.get('slide_pages', [])
    for the_id in id_list:
        parts = the_id.split('-')
        try:
            page_idx = int(parts[0])
            # id可以是任意层级的（page、shape、para任意层级）
            if str(page_idx) == the_id:
                del slide_pages[page_idx]
            else:
                found_it = False
                for s_idx, a_shape in enumerate(slide_pages[page_idx]['slide_data']):
                    if a_shape['id'] == the_id:
                        del slide_pages[page_idx]['slide_data'][s_idx]
                        break
                    elif a_shape['type'] == 'text':
                        for p_idx, a_para in enumerate(a_shape.get('frame_data', [])):
                            if a_para['id'] == the_id:
                                del slide_pages[page_idx]['slide_data'][s_idx]['frame_data'][p_idx]
                                found_it = True
                                break
                    if found_it:
                        break
        except:
            print('id不存在', the_id)
    json.dump(json_data, open(os.path.join(base_config.PPT_DIR, json_fname), 'w', encoding='utf-8'), ensure_ascii=False,
              indent=4)
    return json_data


def update_elements(id_content_list, json_fname):
    json_data = json.load(open(os.path.join(base_config.PPT_DIR, json_fname), encoding='utf-8'))
    slide_pages = json_data.get('slide_pages', [])
    for el_info in id_content_list:
        the_id, the_content = el_info['id'], el_info['content']
        parts = the_id.split('-')
        try:
            found_it = False
            page_idx = int(parts[0])
            for a_shape in slide_pages[page_idx]['slide_data']:  # shape、para实际的idx和id中的不一样，因为解析结束时按位置排序了
                if a_shape['id'] == the_id:
                    a_shape['frame_data'] = the_content if type(the_content) == list else [the_content]
                    break
                elif a_shape['type'] == 'text':
                    for a_para in a_shape.get('frame_data', []):
                        if a_para['id'] == the_id:
                            a_para['data'] = [the_content]
                            found_it = True
                            break
                if found_it:
                    break
        except:
            print('id不存在', the_id)
    json.dump(json_data, open(os.path.join(base_config.PPT_DIR, json_fname), 'w', encoding='utf-8'), ensure_ascii=False,
              indent=4)
    return json_data
