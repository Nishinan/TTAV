import React from 'react';
import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import WebSideBar from './component/webSideBar';
import { AppCombinedView } from './views/plotView';
import { useDefaultStore } from './state/state.unified';
import './index.css';

function RootLayout() {
  const { eifSessionInfo } = useDefaultStore(['eifSessionInfo']);
  const isEifMode = !!eifSessionInfo?.isEifBundle;

  return (
    <div style={{ width: '100%', height: '100%', display: 'flex' }}>
      {!isEifMode && (
        <>
          <div style={{ width: 220, flexShrink: 0, height: '100%', borderRight: '1px solid #ccc', overflow: 'hidden' }}>
            <WebSideBar />
          </div>
        </>
      )}
      <div style={{ flex: 1, minWidth: 0, height: '100%' }}>
        <AppCombinedView />
      </div>
    </div>
  );
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <RootLayout />
  </StrictMode>
);
