/**
 * OCTA 血管网络分析系统 - Electron 预加载脚本
 * 
 * 在渲染进程中安全地暴露 Node.js 功能
 */

const { contextBridge, ipcRenderer } = require('electron');

// 暴露安全的 API 到渲染进程
contextBridge.exposeInMainWorld('electronAPI', {
    // 文件对话框
    openFile: (options) => ipcRenderer.invoke('dialog:openFile', options),
    openFolder: () => ipcRenderer.invoke('dialog:openFolder'),
    saveFile: (options) => ipcRenderer.invoke('dialog:saveFile', options),
    
    // 应用信息
    getAppInfo: () => ipcRenderer.invoke('app:getInfo'),
    
    // 系统操作
    showItemInFolder: (filepath) => ipcRenderer.invoke('shell:showItemInFolder', filepath),
    
    // 平台信息
    platform: process.platform
});
