
# vscode debug 경로 바꾸기

vscode에서 디버깅을 실시하면 기본적으로 루트에서 실행됨.

바꿔주는 코드는 * "cwd": "${fileDirname}"를 추가하면 됨
```
{
    "version": "0.2.0",
    "configurations": [
    {
            "name": "Python: Current File (Integrated Terminal)",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "cwd": "${fileDirname}"
    }, 
}
```

https://stackoverflow.com/questions/38623138/vscode-how-to-set-working-directory-for-debugging-a-python-program

---------------------------------------









