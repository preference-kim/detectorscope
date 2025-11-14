**SAE Dataset Visualizer**

To run the visualizer you must provide a path to the local huggingface dataset
as well as which columns to use. For example:

```bash
python app.py --dataset ../activation/all_activations --text_col response --values_col hook_sae_acts_post
```

When you run the command above, VSCode's integrated terminal will likely show
a pop-up message like:

> Your application is running on `http://localhost:5001`. Do you want to open?

Click **"Open in Browser"**. VSCode will automatically forward the port from
your remote machine to your local machine, and the visualizer will open in a
new tab in your local web browser
