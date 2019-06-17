#!/bin/env python
# -*- coding: utf-8 -*-

# 20180430

import difflib
import sys
import argparse


# 读取建表语句或配置文件
def read_file(file_name):
    try:
        file_desc = open(file_name, 'r')
        # 读取后按行分割
        text = file_desc.read().splitlines()
        file_desc.close()
        return text
    except IOError as error:
        print ('Read input file Error: {0}'.format(error))
        sys.exit()


# 比较两个文件并把结果生成一份html文本
def compare_file(file1, file2):
    if file1 == "" or file2 == "":
        print ('文件路径不能为空：第一个文件的路径：{0}, 第二个文件的路径：{1} .'.format(file1, file2))
        sys.exit()
    else:
        print ("正在比较文件{0} 和 {1}".format(file1, file2))
    text1_lines = read_file(file1)
    text2_lines = read_file(file2)
    diff = difflib.HtmlDiff()    # 创建HtmlDiff 对象
    result = diff.make_file(text1_lines, text2_lines)  # 通过make_file 方法输出 html 格式的对比结果
    # 将结果写入到result_comparation.html文件中
    try:
        with open('result_comparation.html', 'w') as result_file:
            result_file.write(result)
            print ("0==}==========> Successfully Finished\n")
    except IOError as error:
        print ('写入html文件错误：{0}'.format(error))


if __name__ == "__main__":
    # To define two arguments should be passed in, and usage: -f1 fname1 -f2 fname2
    my_parser = argparse.ArgumentParser(description="传入两个文件参数")
    my_parser.add_argument('-f1', action='store', dest='fname1', required=True)
    my_parser.add_argument('-f2', action='store', dest='fname2', required=True)
    # retrieve all input arguments
    given_args = my_parser.parse_args()
    file1 = given_args.fname1
    file2 = given_args.fname2
    compare_file(file1, file2)
