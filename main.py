d={
    "rs":{
        "alpha": 100,
        "max_iter": 10000,
        "score": 0.96
    },
    "ls":{
        "alpha": 100,
        "max_iter": 10000,
        "score": 0.92
    },
    "es":{
        "alpha": 100,
        "max_iter": 10000,
        "score": 0.95
    }
}
best_model = min(d.values(),key=lambda x: x["score"])
print(list(d.keys())[list(d.values()).index(best_model)])