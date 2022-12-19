#!/usr/bin/env python3

from my_solution import *

# 测试用例
def test_solution():
    train()
    result = test()
    
    # 正确答案
    correct_solution = 'i love China'
    
    # 程序求解结果
    assert correct_solution == result
