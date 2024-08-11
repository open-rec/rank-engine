# rank-engine

Online recommend rank service

## start

```shell
bash start.sh
```

## api

Access http://127.0.0.1:8000/docs#/ to get more information.  
eg:  
score   
http://127.0.0.1:8000/model/score
request
```json
{
    "user_id": "test",
    "item_ids":["5105858","3785327","123"]
}
```
response
```json
{
    "code": 0,
    "status": "success",
    "data": {
        "123": 0.0,
        "5105858": 0.6175433993339539,
        "3785327": 0.510399341583252
    },
    "message": ""
}
```

## Model
only support LR now.  
eg:  
Type: LR  
Dimension: 63  
Path: rank-engine/model/lr.pth