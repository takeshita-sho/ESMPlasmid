import re
import pandas as pd
import matplotlib.pyplot as plt


def read_loss_from_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    # Use regular expression to find all floats after "- Loss: "
    loss_values = re.findall(r'- Loss: (\d+\.\d+)', content)

    return list(map(float, loss_values))

def create_dataframe(loss_values):
    df = pd.DataFrame({'Step': range(1, len(loss_values) + 1), 'Loss': loss_values})
    return df

def plot_loss_vs_step(df):
    plt.scatter(df['Step'], df['Loss'], marker='.', s=1, color='cornflowerblue')
    plt.title('Loss vs Step')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.grid(False)
    plt.savefig('/nfshomes/stakeshi/esm/esm/model/graph.png',dpi=800)

def main():
    file_path = '/nfshomes/stakeshi/esm/esm/model/train-1785743.o'  # Replace with the path to your file
    loss_values = read_loss_from_file(file_path)

    if loss_values:
        df = create_dataframe(loss_values)
        plot_loss_vs_step(df)
        print('Graph Saved')
    else:
        print("No loss values found in the file.")

if __name__ == "__main__":
    main()
