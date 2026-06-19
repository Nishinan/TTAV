import { useState } from 'react';
import { ChevronDown } from 'lucide-react';

interface FunctionalBlockProps {
    label?: string;
    children?: null | React.ReactNode | React.ReactNode[];
    /** Collapsed by default. Pass true for secondary/less-used sections. */
    defaultCollapsed?: boolean;
}

export function FunctionalBlock({ label, children, defaultCollapsed = false }: FunctionalBlockProps) {
    const [collapsed, setCollapsed] = useState(defaultCollapsed);

    return (
        <div className="functional-block" style={{ overflow: 'visible' }}>
            {label && (
                <div
                    className="functional-block-title"
                    onClick={() => setCollapsed(c => !c)}
                    role="button"
                    aria-expanded={!collapsed}
                >
                    <span>{label}</span>
                    <ChevronDown
                        size={12}
                        className={`collapse-chevron${collapsed ? ' collapsed' : ''}`}
                    />
                </div>
            )}
            <div
                className={`functional-block-body${collapsed ? ' collapsed' : ''}`}
                style={{ maxHeight: collapsed ? 0 : 2000 }}
            >
                {children}
            </div>
        </div>
    );
}

export function ComponentBlock(props: { label?: string; children?: null | React.ReactNode | React.ReactNode[]; }) {
    return (
        <div className="component-block">
            {props.label && <div className="label">{props.label}</div>}
            {props.children}
        </div>
    );
}
