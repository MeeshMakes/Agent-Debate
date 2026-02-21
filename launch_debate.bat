@echo off
title Agent Debate System — Astra vs Nova
cd /d "c:\Users\FathomPC\Desktop\Agent-Debate\agent_debate_nextgen"
set PYTHONPATH=c:\Users\FathomPC\Desktop\Agent-Debate\agent_debate_nextgen
"C:\Users\FathomPC\AppData\Local\Programs\Python\Python314\python.exe" -m app.main
if errorlevel 1 (
    echo.
    echo  ERROR — the app crashed. Check the output above.
    pause
)
