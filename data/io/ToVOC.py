from xml.dom.minidom import Document
import cv2
import os
import glob
import shutil
import numpy as np

def generate_xml(name, lables, img_size, class_set):
    doc = Document()

    def append_xml_node_attr(child, parent=None, text=None):
        ele = doc.createElement(child)
        if not text is None:
            text_node = doc.createTextNode(text)
            ele.appendChild(text_node)
        parent = doc if parent is None else parent
        parent.appendChild(ele)
        return ele

    img_name = name + '.jpg'
    # create header
    annotation = append_xml_node_attr('annotation')
    append_xml_node_attr('folder', parent=annotation, text='text')
    append_xml_node_attr('filename', parent=annotation, text=img_name)
    source = append_xml_node_attr('source', parent=annotation)
    append_xml_node_attr('database', parent=source, text='ali_text_database')
    append_xml_node_attr('annotation', parent=source, text='text')
    append_xml_node_attr('image', parent=source, text='text')
    append_xml_node_attr('flickrid', parent=source, text='000000')
    owner = append_xml_node_attr('owner', parent=annotation)
    append_xml_node_attr('name', parent=owner, text='yang')
    append_xml_node_attr('flickrid', parent=owner, text='000000')
    size = append_xml_node_attr('size', annotation)
    append_xml_node_attr('width', size, str(img_size[1]))
    append_xml_node_attr('height', size, str(img_size[0]))
    append_xml_node_attr('depth', size, str(img_size[2]))
    append_xml_node_attr('segmented', parent=annotation, text='0')

    # create objects
    objs = []
    for lable in lables:
        splitted_line = lable.strip().lower().split(',')
        obj = append_xml_node_attr('object', parent=annotation)


        x0, y0, x1, y1, x2, y2, x3, y3 = int(float(splitted_line[0]) + 1), int(float(splitted_line[1]) + 1),\
                                         int(float(splitted_line[2]) + 1), int(float(splitted_line[3]) + 1),\
                                         int(float(splitted_line[4]) + 1), int(float(splitted_line[5]) + 1),\
                                         int(float(splitted_line[6]) + 1), int(float(splitted_line[7]) + 1)


        append_xml_node_attr('name', parent=obj, text=class_set)
        append_xml_node_attr('pose', parent=obj, text='Unknown')
        append_xml_node_attr('truncated', parent=obj, text='1')
        append_xml_node_attr('difficult', parent=obj, text='0')
        bb = append_xml_node_attr('bndbox', parent=obj)
        append_xml_node_attr('x0', parent=bb, text=str(x0))
        append_xml_node_attr('y0', parent=bb, text=str(y0))
        append_xml_node_attr('x1', parent=bb, text=str(x1))
        append_xml_node_attr('y1', parent=bb, text=str(y1))
        append_xml_node_attr('x2', parent=bb, text=str(x2))
        append_xml_node_attr('y2', parent=bb, text=str(y2))
        append_xml_node_attr('x3', parent=bb, text=str(x3))
        append_xml_node_attr('y3', parent=bb, text=str(y3))

        # o = {'class': cls, 'box': np.asarray([x1, y1, x2, y2], dtype=float), \
        #      'truncation': truncation, 'difficult': difficult, 'occlusion': occlusion}
        # objs.append(o)
    return doc

    # return doc, objs


# def _is_hard(cls, truncation, occlusion, x1, y1, x2, y2):
#     hard = False
#     if y2 - y1 < 25 and occlusion >= 2:
#         hard = True
#         return hard
#     if occlusion >= 3:
#         hard = True
#         return hard
#     if truncation > 0.8:
#         hard = True
#         return hard
#     return hard


def build_voc_dirs(outdir):
    mkdir = lambda dir: os.makedirs(dir) if not os.path.exists(dir) else None
    mkdir(outdir)
    mkdir(os.path.join(outdir, 'Annotations'))
    mkdir(os.path.join(outdir, 'JPEGImages'))

    return os.path.join(outdir, 'Annotations'), os.path.join(outdir, 'JPEGImages')


if __name__ == '__main__':
    _imagedir = '/Users/yangcd/tensorflow/project/yang_data/image'
    _labeldir = '/Users/yangcd/tensorflow/project/yang_data/label'
    _outdir = 'VOCdevkit/VOCdevkit_train'
    _annotationsdir, _jpegfir = build_voc_dirs(_outdir)
    class_set = 'text'
    files = os.listdir(_imagedir)
    files.sort()

    for file in files:
        path, basename = os.path.split(file)
        stem, ext = os.path.splitext(basename)
        img_file = os.path.join(_imagedir, stem + '.jpg')
        print(img_file)
        img = cv2.imread(img_file)
        img_size = img.shape
        if not img_size: continue
        lable_file = os.path.join(_labeldir, stem + '.txt')

        with open(lable_file, 'r') as f:
            lables = f.readlines()
        doc = generate_xml(stem, lables, img_size, class_set)

        xmlfile = os.path.join(_annotationsdir, stem + '.xml')
        with open(xmlfile, 'w') as f:
            f.write(doc.toprettyxml(indent='	'))
        cv2.imwrite(os.path.join(_jpegfir, stem + '.jpg'), img)
