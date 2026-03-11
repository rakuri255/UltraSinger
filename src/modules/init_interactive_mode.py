from rich.console import Console  
from rich.table import Table
from Settings import Settings  
from modules.Audio.separation import DemucsModel
from modules.Speech_Recognition.Whisper import WhisperModel
from modules.DeviceDetection.device_detection import check_gpu_support

import os  
  
def get_input_file(console, settings, header):  
    while True:  
        input_file = console.input(f"{header} Enter the path to the input file ([green]audio file[/green], [green]Ultrastar txt[/green], or [green]YouTube URL[/green]): ").strip()  
        if input_file:  
            settings.input_file_path = input_file 
            print(settings.input_file_path) 
            break  
        else:  
            console.print(f"{header} [bold red]Error:[/bold red] Input file cannot be empty. Please try again.\n")  
  
def get_output_folder(console, settings, header):  
    output_folder = console.input(f"{header} Enter the output folder path (leave empty for default '[green]output[/green]' folder): ").strip()  
    if output_folder:  
        settings.output_folder_path = output_folder  
    else:  
        dirname = os.getcwd() if settings.input_file_path.startswith("https:") else os.path.dirname(settings.input_file_path)  
        settings.output_folder_path = os.path.join(dirname, "output")  
  
def select_model(console, header, model_enum, model_type, default_model):  
    models = [model.value for model in model_enum]  
    console.print(f"\n{header} [bold underline]Available {model_type} Models:[/bold underline]\n")  
  
    num_columns = 4  
    table = Table(show_header=False, show_edge=False, padding=(0, 2))  
    for _ in range(num_columns):  
        table.add_column()  
  
    items = [f"[bright_green]{idx}.[/bright_green] {model_name}" for idx, model_name in enumerate(models, start=1)]  
    rows = [items[i:i + num_columns] for i in range(0, len(items), num_columns)]  
    for row in rows:  
        row += [""] * (num_columns - len(row))  
        table.add_row(*row)  
    console.print(table)  
  
    while True:  
        choice = console.input(f"\n{header} Enter the [green]{model_type} model[/green] number corresponding to your choice (1-{len(models)}), or leave empty for default ([cyan]{default_model.value}[/cyan]): ").strip()  
        if not choice:  
            return default_model  
        elif choice.isdigit() and 1 <= int(choice) <= len(models):  
            return model_enum(models[int(choice) - 1])  
        else:  
            console.print(f"{header} [bold red]Error:[/bold red] Invalid choice. Please select a valid number.\n")  
  
def configure_additional_options(console, settings, header):  
    additional_options_input = console.input(  
        f"\n{header} Do you want to configure [green]additional options[/green]? "  
        f"([bright_green]y[/bright_green]/[bright_red]n[/bright_red], default '[bright_red]n[/bright_red]'): "  
    ).strip().lower()  
  
    if additional_options_input == 'y':  
        console.print(f"\n{header} [bold underline]Additional options:[/bold underline]\n")  

        # Whisper Batch Size  
        whipser_batch_size_response = console.input(  
            f"{header} Enter the [green]Whisper batch size[/green] (default [cyan]16[/cyan]): "  
        ).strip()  
        settings.whisper_batch_size = int(whipser_batch_size_response) if whipser_batch_size_response.isdigit() else 16  
  
        # Whisper Compute Type  
        whisper_compute_choice = console.input(
            f"{header} Enter the [green]Whisper compute type[/green] (default '[cyan]float16[/cyan]' for CUDA and '[cyan]int8[/cyan]' for CPU): "  
        ).strip()  
        if whisper_compute_choice:  
            settings.whisper_compute_type = whisper_compute_choice  
  
        # Create Plot  
        settings.create_plot = console.input(  
            f"{header} Create [green]plot[/green]? ([bright_green]y[/bright_green]/[bright_red]n[/bright_red], default '[bright_red]n[/bright_red]'): "  
        ).strip().lower() == 'y'  
  
        # Create MIDI  
        settings.create_midi = not (console.input(  
            f"{header} Create [green]MIDI file[/green]? ([bright_green]y[/bright_green]/[bright_red]n[/bright_red], default '[bright_green]y[/bright_green]'): "  
        ).strip().lower() == 'n')  
  
        # Disable Hyphenation  
        settings.hyphenation = not (console.input(  
            f"{header} Disable [green]hyphenation[/green]? ([bright_green]y[/bright_green]/[bright_red]n[/bright_red], default '[bright_red]n[/bright_red]'): "  
        ).strip().lower() == 'y')  
  
        # Disable Vocal Separation  
        settings.use_separated_vocal = not (console.input(  
            f"{header} Disable [green]vocal separation[/green]? ([bright_green]y[/bright_green]/[bright_red]n[/bright_red], default '[bright_red]n[/bright_red]'): "  
        ).strip().lower() == 'y')  
  
        # Disable Karaoke Creation  
        settings.create_karaoke = not (console.input(  
            f"{header} Disable [green]karaoke creation[/green]? ([bright_green]y[/bright_green]/[bright_red]n[/bright_red], default '[bright_red]n[/bright_red]'): "  
        ).strip().lower() == 'y')  
  
        # Create Audio Chunks  
        settings.create_audio_chunks = console.input(  
            f"{header} Create [green]audio chunks[/green]? ([bright_green]y[/bright_green]/[bright_red]n[/bright_red], default '[bright_red]n[/bright_red]'): "  
        ).strip().lower() == 'y'  
  
        # Ignore Audio  
        settings.ignore_audio = console.input(  
            f"{header} Ignore [green]audio[/green] and use Ultrastar txt only? ([bright_green]y[/bright_green]/[bright_red]n[/bright_red], default '[bright_red]n[/bright_red]'): "  
        ).strip().lower() == 'y'  
  
        # Force CPU Usage  
        settings.force_cpu = console.input(  
            f"{header} Force [green]CPU usage[/green]? ([bright_green]y[/bright_green]/[bright_red]n[/bright_red], default '[bright_red]n[/bright_red]'): "  
        ).strip().lower() == 'y'  
  
        # Device settings based on CPU usage  
        if settings.force_cpu:  
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  
        else:  
            settings.tensorflow_device, settings.pytorch_device = check_gpu_support()  
  
        # Force Whisper CPU Usage  
        settings.force_whisper_cpu = console.input(  
            f"{header} Force [green]Whisper CPU usage[/green]? ([bright_green]y[/bright_green]/[bright_red]n[/bright_red], default '[bright_red]n[/bright_red]'): "  
        ).strip().lower() == 'y'  
  
        # Force Crepe CPU Usage  
        settings.force_crepe_cpu = console.input(  
            f"{header} Force [green]Crepe CPU usage[/green]? ([bright_green]y[/bright_green]/[bright_red]n[/bright_red], default '[bright_red]n[/bright_red]'): "  
        ).strip().lower() == 'y'  
  
        # Keep Cache  
        settings.keep_cache = console.input(  
            f"{header} Keep [green]cache[/green] after execution? ([bright_green]y[/bright_green]/[bright_red]n[/bright_red], default '[bright_red]n[/bright_red]'): "  
        ).strip().lower() == 'y'  
  
        # Language  
        language = console.input(  
            f"\n{header} Enter the [green]language code[/green] (e.g., '[cyan]en[/cyan]' for English, '[cyan]es[/cyan]' for Spanish) or leave empty for [cyan]auto-detect[/cyan]: "  
        ).strip()  
        if language:  
            settings.language = language  
  
        # Transcribe Numbers as Numerics  
        keep_numbers_input = console.input(  
            f"\n{header} Do you want to transcribe [green]numbers as numerics[/green]? ([bright_green]y[/bright_green]/[bright_red]n[/bright_red], default '[bright_red]n[/bright_red]'): "  
        ).strip().lower()  
        settings.keep_numbers = keep_numbers_input == 'y'  
  
        # MuseScore Path  
        musescore_path = console.input(  
            f"\n{header} Enter the path to [green]MuseScore executable[/green] for sheet generation (leave empty to skip): "  
        ).strip()  
        if musescore_path:  
            settings.musescore_path = musescore_path  
  
        # Cookie File for YouTube Downloads  
        cookie_file = console.input(  
            f"\n{header} Enter the path to [green]cookies.txt[/green] file (if required for YouTube downloads, leave empty otherwise): "  
        ).strip()  
        if cookie_file:  
            settings.cookiefile = cookie_file

        # FFmpeg executable path
        ffmpeg_path = console.input(
            f"\n{header} Enter the path to [green]ffmpeg[/green] executable folder (leave empty for default): "
        ).strip()
        if ffmpeg_path:
            settings.ffmpeg_path = ffmpeg_path
  
def init_settings_interactive(settings: Settings) -> Settings:  
    ULTRASINGER_HEAD = "[bold green][UltraSinger][/bold green]"  
    console = Console()  
    console.print(f"{ULTRASINGER_HEAD} [yellow]UltraSinger Interactive Mode[/yellow]\n")     
    get_input_file(console, settings, ULTRASINGER_HEAD)
    get_output_folder(console, settings, ULTRASINGER_HEAD)  
    settings.whisper_model = select_model(console, ULTRASINGER_HEAD, WhisperModel, "Whisper", WhisperModel.LARGE_V2)  
    settings.demucs_model = select_model(console, ULTRASINGER_HEAD, DemucsModel, "Demucs", DemucsModel.HTDEMUCS)  
  
    configure_additional_options(console, settings, ULTRASINGER_HEAD)  
  
    console.print(f"\n{ULTRASINGER_HEAD} [bold cyan]Thank you! Starting processing...[/bold cyan]\n")  
    return settings  