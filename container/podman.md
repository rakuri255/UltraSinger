# Containerized UltraSinger (Podman)

1. prerequisites
   1. Podman installed (for Windows WSL2 machine provider is recommended) -> [installation instructions](https://podman-desktop.io/docs/installation)
   1. (optional) for GPU acceleration Nvidia Container Toolkit installed -> [GPU container access](https://podman-desktop.io/docs/podman/gpu)
1. clone this repository:
    ```commandline
    git clone https://github.com/rakuri255/UltraSinger.git
    ```
1. build or pull the container image 
    ```bash
    podman build -t ultrasinger .
    # or
    podman pull ghcr.io/rakuri255/ultrasinger:<version> # replace <version> with the desired version or 'latest'
    ```
1. run the container (note that the first time you run the container models will be downloaded which may take a while)
   1. PowerShell:
      ```powershell
      # remember to replace <desired-output-folder> below
      podman run `
        --rm -it --name ultrasinger `
        --device nvidia.com/gpu=all `
        -v $env:USERPROFILE\.cache:/app/UltraSinger/src/.cache `
        -v <desired-output-folder>:/app/UltraSinger/src/output `
        ghcr.io/rakuri255/ultrasinger `
        python3 UltraSinger.py -i <refer to top-level README.md>
      
      # explanation:
      podman run `
        --rm -it --name ultrasinger ` # remove container after run, interactive mode, name the container
        --device nvidia.com/gpu=all ` # optional, enables GPU acceleration if available, requires step 1.ii
        -v $env:USERPROFILE\.cache:/app/UltraSinger/src/.cache ` # cache directory for models
        -v <desired-output-folder>:/app/UltraSinger/src/output ` # output directory
        ghcr.io/rakuri255/ultrasinger ` # latest container image, set to 'ultrasinger' to use the locally built image
        python3 UltraSinger.py -i <refer to top-level README.md> # run UltraSinger, refer to top-level README.md for all options
      ```
   1. Bash:
      ```bash
      # remember to replace <desired-output-folder> below
      podman run \
        --rm -it --name ultrasinger \
        --device nvidia.com/gpu=all \
        -v $HOME/.cache:/app/UltraSinger/src/.cache \
        -v <desired-output-folder>:/app/UltraSinger/src/output \
        ultrasinger \
        python3 UltraSinger.py -i <refer to top-level README.md>
   1. to use cookies for YouTube downloads, you can mount your cookies.txt file into the container:
      ```powershell
      podman run `
        ... ` # same as above
        -v <path-to-cookies.txt>:/app/UltraSinger/src/cookies.txt `
        ultrasinger `
        python3 UltraSinger.py --cookiefile cookies.txt -i <refer to top-level README.md>
      ```
