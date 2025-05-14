def findDecision(obj): #obj[0]: sepal_length, obj[1]: sepal_width, obj[2]: petal_length, obj[3]: petal_width
   # {"feature": "petal_length", "instances": 105, "metric_value": 1.585, "depth": 1}
   if obj[2]>1.996315930098005:
      # {"feature": "petal_width", "instances": 70, "metric_value": 1.0, "depth": 2}
      if obj[3]<=1.6:
         # {"feature": "sepal_length", "instances": 35, "metric_value": 0.1872, "depth": 3}
         if obj[0]<=6.2:
            return '1'
         elif obj[0]>6.2:
            # {"feature": "sepal_width", "instances": 12, "metric_value": 0.4138, "depth": 4}
            if obj[1]>2.8:
               return '1'
            elif obj[1]<=2.8:
               return '1'
