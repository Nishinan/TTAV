import React, { useEffect, useState } from 'react';
import { Input, Select, Radio, Button, Space, Divider } from 'antd';

const DEFAULT_CONTENT_PATH = '/root/project/Dataset/backdoor';
const LOCAL_CONTENT_PATH_CONFIG_URL = '/content-path.local.json';

const methodOptions = [
  { label: 'DVI', value: 'DVI' },
  { label: 'TimeVis', value: 'TimeVis' },
  { label: 'DynaVis', value: 'DynaVis' },
  { label: 'UMAP', value: 'UMAP' },
  { label: 'T-SNE', value: 'T-SNE' },
];

export default function WebSideBar() {
  const [contentPath, setContentPath] = useState(DEFAULT_CONTENT_PATH);
  const [visualizationID, setVisualizationID] = useState('1');
  const [visualizationMethod, setVisualizationMethod] = useState('TimeVis');
  const [dataType, setDataType] = useState<'Image' | 'Text'>('Image');
  const [taskType, setTaskType] = useState<'Classification' | 'Alignment'>('Classification');

  useEffect(() => {
    let cancelled = false;

    const loadLocalContentPath = async () => {
      try {
        const response = await fetch(LOCAL_CONTENT_PATH_CONFIG_URL, { cache: 'no-store' });
        if (!response.ok) return;

        const data = await response.json();
        const localPath = typeof data?.contentPath === 'string' ? data.contentPath.trim() : '';
        if (!cancelled && localPath) {
          setContentPath(localPath);
        }
      } catch {
        // Missing local override is expected in shared/server environments.
      }
    };

    void loadLocalContentPath();
    return () => {
      cancelled = true;
    };
  }, []);

  const startVisualizing = () => {
    window.postMessage(
      {
        command: 'startVisualizing',
        data: {
          contentPath,
          visualizationMethod,
          visualizationID,
          dataType,
          taskType,
          visConfig: {},
        },
      },
      '*'
    );
  };

  const loadVisualization = () => {
    window.postMessage(
      {
        command: 'loadVisualization',
        // data: {
        //   visualizationID,
        //   config: {
        //     contentPath,
        //     visualizationMethod,
        //     dataType,
        //     taskType,
        //   },
        data: {
          contentPath,
          visualizationMethod,
          visualizationID,
          dataType,
          taskType,
          visConfig: {},
        },
      },
      '*'
    );
  };
  
  const syncSession = () => {
    window.postMessage(
      {
        command: 'syncSession',
        // data: {
        //   visualizationID,
        //   config: {
        //     contentPath,
        //     visualizationMethod,
        //     dataType,
        //     taskType,
        //   },
        data: {
          contentPath,
          visualizationMethod,
          visualizationID,
          dataType,
          taskType,
          visConfig: {},
        },
      },
      '*'
    );
  };
  return (
    <div style={{ width: '100%', height: '100%', padding: 12, display: 'flex', flexDirection: 'column', gap: 12 }}>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        <span style={{ fontSize: 12, fontWeight: 600 }}>Content Path</span>
        <Input value={contentPath} onChange={(e) => setContentPath(e.target.value)} placeholder="/path/to/content" />
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        <span style={{ fontSize: 12, fontWeight: 600 }}>Visualization ID</span>
        <Input value={visualizationID} onChange={(e) => setVisualizationID(e.target.value)} placeholder="session-id" />
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        <span style={{ fontSize: 12, fontWeight: 600 }}>Method</span>
        <Select style={{ width: '100%' }} value={visualizationMethod} onChange={(v) => setVisualizationMethod(v)} options={methodOptions} size="middle" />
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        <span style={{ fontSize: 12, fontWeight: 600 }}>Data Type</span>
        <Radio.Group value={dataType} onChange={(e) => setDataType(e.target.value)}>
          <Space direction="vertical">
            <Radio value={'Image'}>Image</Radio>
            <Radio value={'Text'}>Text</Radio>
          </Space>
        </Radio.Group>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        <span style={{ fontSize: 12, fontWeight: 600 }}>Task Type</span>
        <Radio.Group value={taskType} onChange={(e) => setTaskType(e.target.value)}>
          <Space direction="vertical">
            <Radio value={'Classification'}>Classification</Radio>
            <Radio value={'Alignment'}>Alignment</Radio>
          </Space>
        </Radio.Group>
      </div>
      <Divider style={{ margin: '8px 0' }} />
      <div style={{ display: 'flex', gap: 8 }}>
        <Button type="primary" onClick={startVisualizing} style={{ flex: 1 }}>Start Visualizing</Button>
        <Button onClick={loadVisualization} style={{ flex: 1 }}>Load Visualization</Button>
         </div>
        <Button onClick={syncSession} style={{ width: '100%', borderStyle: 'dashed' }}>
    Sync Backend Session
  </Button>
     
    </div>
  );
}
