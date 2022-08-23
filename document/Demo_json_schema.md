## JSON Schema For Demo
  1. Request

  ```json
    {
       "date": "yyyy-mm-dd_hh:mm:ss.msec",
       "sentences": [
        {"id": "str or integer", "text": "str"}
       ]
    }
  ```
  
  2. Response
  
  ```json
    {
      "date": "yyyy-mm-dd_hh:mm:ss.msec",
      "results": [{
        "id": "str", 
        "text": "str",
        "ne": [
          {"word": "str", "label": "str", "begin": "integer", "end": "integer"}
        ]
      }]
    }
  ```
