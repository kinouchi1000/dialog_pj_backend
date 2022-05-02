# dialog project backend

NTTのjapanese dialog transformerを用いた、対話システムのプロジェクト

# requirement

- docker 
- python 

# Quick start

1. Let's build image and run container
   
   ```bash
   cd docker
   docker compose build 
   docker compose up -d
   ```

   You can see the logs with below.
   ```bash
   docker compose logs -f 
   ```

2. Install some libraries
   
   ```bash
   cd ../
   pip install -r requirement.txt
   ```

3. Please run test code
   
    ```bash
    python test/client_sample/main.py
    ```

    You can see result like below.
    ```bash
    ❯ python test/client_sample/main.py
    You sent hello!!! right?
    ['History4', 'History5', 'History6', 'History7', 'History8', 'History9', 'History10', 'History11', 'History12', 'History13']
    ['History1', 'History2', 'History3', 'History4', 'History5', 'History6', 'History7', 'History8', 'History9', 'History10', 'History11', 'History12', 'History13', 'History14', 'hello!!!', 'You sent hello!!! right?', 'hello!!!', 'You sent hello!!! right?', 'hello!!!', 'You sent hello!!! right?']
    ```
