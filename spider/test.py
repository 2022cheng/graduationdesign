import json

# 原始字符串数据
data = "['100', '499']"

# 转换字符串为列表
company_people_list = json.loads(data.replace("'", "\""))  # 先替换单引号为双引号，json.loads才会正确解析

# 转换为整数类型
company_people_list = [int(x) for x in company_people_list]

print(company_people_list)
