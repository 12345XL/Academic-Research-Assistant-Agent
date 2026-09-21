import type { SVGProps } from 'react';

export type IconName = 'book' | 'search' | 'arrow' | 'database' | 'layers' | 'file' | 'chevron' | 'download' | 'refresh' | 'close' | 'check' | 'external' | 'info' | 'clock';

const paths: Record<IconName, React.ReactNode> = {
  book: <><path d="M3 4h6l3 2 3-2h6v15h-6l-3 2-3-2H3z"/><path d="M12 6v15"/></>,
  search: <><circle cx="10.5" cy="10.5" r="6.5"/><path d="m16 16 4.5 4.5"/></>,
  arrow: <><path d="M4 12h16m-6-6 6 6-6 6"/></>,
  database: <><ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v7c0 4 16 4 16 0V5M4 12v7c0 4 16 4 16 0v-7"/></>,
  layers: <><path d="m12 3 9 5-9 5-9-5 9-5zm-9 9 9 5 9-5m-18 5 9 5 9-5"/></>,
  file: <><path d="M14 3H5v18h14V8zM14 3v6h5M8 13h8m-8 4h5"/></>,
  chevron: <path d="m9 5 7 7-7 7"/>,
  download: <><path d="M12 3v12m-5-5 5 5 5-5M4 16v5h16v-5"/></>,
  refresh: <><path d="M20 7a8 8 0 1 0 0 10M20 3v5h-5"/></>,
  close: <path d="m6 6 12 12M6 18 18 6"/>,
  check: <path d="m5 12 4 4L19 6"/>,
  external: <><path d="M14 3h7v7M21 3l-12 12M11 4H4v16h16v-7"/></>,
  info: <><circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7h.01"/></>,
  clock: <><circle cx="12" cy="12" r="9"/><path d="M12 6v6l4 2"/></>,
};

export default function Icon({ name, ...props }: SVGProps<SVGSVGElement> & { name: IconName }) {
  return <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" {...props}>{paths[name]}</svg>;
}
