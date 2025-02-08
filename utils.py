import sys




def print_progress_bar(epoch, iteration, total, length=50):
    progress = int(length * iteration / total)
    if epoch != -1:
        bar = f"\033[31m Epoch {epoch}:\033[97m [{'=' * progress}{' ' * (length - progress)}] {int(100 * iteration / total)}%"
    else:
        bar = f"[{'=' * progress}{' ' * (length - progress)}] {int(100 * iteration / total)}%"
    sys.stdout.write(f"\r{bar}")
    sys.stdout.flush()