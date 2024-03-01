from dvclive import Live

NUM_EPOCHS = 10

def train_model():
    pass

def evaluate_model():
    pass

with Live() as live:
    live.log_param("epochs", NUM_EPOCHS)

    for epoch in range(NUM_EPOCHS):
        train_model(...)
        metrics = evaluate_model(...)
        for metric_name, value in metrics.items():
            live.log_metric(metric_name, value)
        live.next_step()

    live.log_artifact("model.pkl", type="model")