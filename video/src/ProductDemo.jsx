import React from 'react';
import {
  AbsoluteFill,
  Easing,
  interpolate,
  spring,
  useCurrentFrame,
  useVideoConfig,
} from 'remotion';
import {TransitionSeries, linearTiming} from '@remotion/transitions';
import {fade} from '@remotion/transitions/fade';

const palette = {
  bg: '#020303',
  bg2: '#0a0b0d',
  text: '#f5f5f7',
  muted: 'rgba(245,245,247,0.62)',
  mutedStrong: 'rgba(245,245,247,0.78)',
  card: 'rgba(255,255,255,0.045)',
  cardStrong: 'rgba(255,255,255,0.08)',
  border: 'rgba(255,255,255,0.12)',
  borderSoft: 'rgba(255,255,255,0.08)',
  blue: '#7fb2ff',
  blueSoft: 'rgba(127,178,255,0.18)',
  green: '#52d18f',
  amber: '#f6c760',
  red: '#ff7c70',
};

const shellStyle = {
  fontFamily:
    '"SF Pro Display", "SF Pro Text", ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, sans-serif',
  color: palette.text,
  backgroundColor: palette.bg,
};

const modelList = [
  'ModernBERT Sentiment',
  'Toxic-BERT',
  'BART MNLI',
  'GoEmotions',
  'Twitter RoBERTa Irony',
  'VADER',
];

const textComment = 'smh worst update ever, nothing works anymore';

const batchRows = [
  ['This app is amazing, it solved all my problems!', 'Positive', 'Praise', '94%'],
  ['smh worst update ever, nothing works anymore', 'Negative', 'Complaint', '97%'],
  ['How do I export my data to CSV?', 'Neutral', 'Question', '86%'],
];

const rise = (frame, fps, delay = 0, duration = 26) =>
  spring({
    frame: frame - delay,
    fps,
    durationInFrames: duration,
    config: {damping: 200, stiffness: 120},
  });

const drift = (frame, start, end, from, to) =>
  interpolate(frame, [start, end], [from, to], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
    easing: Easing.out(Easing.cubic),
  });

const fadeUpStyle = (frame, fps, delay = 0, y = 30) => {
  const p = rise(frame, fps, delay);
  return {
    opacity: p,
    transform: `translateY(${interpolate(p, [0, 1], [y, 0])}px)`,
  };
};

const SceneBackground = ({frame, children, accentX = 0.2, accentY = 0.15}) => (
  <AbsoluteFill
    style={{
      background: [
        `radial-gradient(circle at ${accentX * 100}% ${accentY * 100}%, rgba(127,178,255,0.16), transparent 24%)`,
        'radial-gradient(circle at 85% 18%, rgba(255,255,255,0.06), transparent 18%)',
        'linear-gradient(180deg, #040506 0%, #0b0d10 46%, #030405 100%)',
      ].join(', '),
    }}
  >
    <div
      style={{
        position: 'absolute',
        inset: 0,
        opacity: 0.4,
        backgroundImage:
          'linear-gradient(rgba(255,255,255,0.025) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.025) 1px, transparent 1px)',
        backgroundSize: '120px 120px',
        transform: `translateY(${drift(frame, 0, 120, 0, -12)}px)`,
      }}
    />
    <div
      style={{
        position: 'absolute',
        inset: 0,
        background:
          'radial-gradient(circle at center, transparent 56%, rgba(0,0,0,0.45) 100%)',
      }}
    />
    {children}
  </AbsoluteFill>
);

const Panel = ({children, style}) => (
  <div
    style={{
      background:
        'linear-gradient(180deg, rgba(255,255,255,0.08) 0%, rgba(255,255,255,0.035) 100%)',
      border: `1px solid ${palette.borderSoft}`,
      borderRadius: 34,
      boxShadow:
        '0 28px 80px rgba(0,0,0,0.38), inset 0 1px 0 rgba(255,255,255,0.08)',
      backdropFilter: 'blur(18px)',
      ...style,
    }}
  >
    {children}
  </div>
);

const Kicker = ({children}) => (
  <div
    style={{
      fontSize: 16,
      letterSpacing: '0.28em',
      textTransform: 'uppercase',
      color: palette.muted,
      fontWeight: 700,
      marginBottom: 18,
    }}
  >
    {children}
  </div>
);

const BrowserFrame = ({children, frame}) => {
  const scale = interpolate(frame, [0, 120], [0.965, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
    easing: Easing.out(Easing.cubic),
  });

  return (
    <div
      style={{
        width: 1400,
        height: 790,
        padding: 20,
        borderRadius: 40,
        background:
          'linear-gradient(180deg, rgba(255,255,255,0.08) 0%, rgba(255,255,255,0.02) 100%)',
        border: `1px solid ${palette.border}`,
        boxShadow: '0 40px 120px rgba(0,0,0,0.45)',
        transform: `scale(${scale})`,
      }}
    >
      <div
        style={{
          display: 'flex',
          gap: 8,
          alignItems: 'center',
          padding: '4px 6px 18px',
        }}
      >
        {['rgba(255,255,255,0.18)', 'rgba(255,255,255,0.16)', 'rgba(255,255,255,0.14)'].map(
          (dot) => (
            <div
              key={dot}
              style={{
                width: 10,
                height: 10,
                borderRadius: '50%',
                background: dot,
              }}
            />
          ),
        )}
        <div
          style={{
            marginLeft: 14,
            color: palette.muted,
            fontSize: 15,
            letterSpacing: '0.03em',
          }}
        >
          smart-comment-classification.app
        </div>
      </div>
      <div
        style={{
          width: '100%',
          height: 722,
          borderRadius: 30,
          background:
            'linear-gradient(180deg, rgba(6,7,9,0.94) 0%, rgba(13,15,18,0.98) 100%)',
          overflow: 'hidden',
          position: 'relative',
        }}
      >
        {children}
      </div>
    </div>
  );
};

const MetricChip = ({label, value, color}) => (
  <div
    style={{
      padding: '16px 18px',
      borderRadius: 22,
      background: palette.card,
      border: `1px solid ${palette.borderSoft}`,
      minWidth: 180,
    }}
  >
    <div
      style={{
        fontSize: 12,
        textTransform: 'uppercase',
        letterSpacing: '0.12em',
        color: palette.muted,
      }}
    >
      {label}
    </div>
    <div
      style={{
        marginTop: 8,
        fontSize: 24,
        fontWeight: 700,
        color: color ?? palette.text,
        letterSpacing: '-0.03em',
      }}
    >
      {value}
    </div>
  </div>
);

const ConfidenceRow = ({label, pct, color}) => (
  <div
    style={{
      display: 'grid',
      gridTemplateColumns: '130px 1fr 56px',
      gap: 16,
      alignItems: 'center',
    }}
  >
    <div
      style={{
        fontSize: 14,
        textTransform: 'uppercase',
        letterSpacing: '0.12em',
        color,
        fontWeight: 700,
      }}
    >
      {label}
    </div>
    <div
      style={{
        height: 8,
        borderRadius: 999,
        background: 'rgba(255,255,255,0.07)',
        overflow: 'hidden',
      }}
    >
      <div
        style={{
          height: '100%',
          width: `${pct}%`,
          borderRadius: 999,
          background: color,
        }}
      />
    </div>
    <div
      style={{
        fontSize: 14,
        color: palette.muted,
        textAlign: 'right',
        fontVariantNumeric: 'tabular-nums',
      }}
    >
      {pct}%
    </div>
  </div>
);

const IntroScene = ({title, subtitle}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const titleProgress = Math.floor(
    interpolate(frame, [10, 72], [0, title.length], {
      extrapolateLeft: 'clamp',
      extrapolateRight: 'clamp',
    }),
  );

  return (
    <AbsoluteFill style={shellStyle}>
      <SceneBackground frame={frame}>
        <AbsoluteFill style={{padding: '92px 96px'}}>
          <div style={{...fadeUpStyle(frame, fps, 0), maxWidth: 760}}>
            <Kicker>Product Demo</Kicker>
            <div
              style={{
                fontSize: 96,
                lineHeight: 0.93,
                letterSpacing: '-0.075em',
                fontWeight: 700,
              }}
            >
              {title.slice(0, titleProgress)}
              <span style={{opacity: frame % 18 < 9 ? 1 : 0, color: palette.blue}}>|</span>
            </div>
            <div
              style={{
                marginTop: 26,
                fontSize: 30,
                lineHeight: 1.33,
                color: palette.mutedStrong,
                maxWidth: 660,
              }}
            >
              {subtitle}
            </div>
          </div>

          <div style={{...fadeUpStyle(frame, fps, 18, 40), marginTop: 70}}>
            <BrowserFrame frame={frame}>
              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns: '1.1fr 0.9fr',
                  gap: 24,
                  padding: 28,
                  height: '100%',
                }}
              >
                <Panel
                  style={{
                    padding: 34,
                    display: 'flex',
                    flexDirection: 'column',
                    justifyContent: 'space-between',
                  }}
                >
                  <div>
                    <div style={{fontSize: 18, color: palette.muted, marginBottom: 16}}>
                      Single comments or full datasets
                    </div>
                    <div
                      style={{
                        fontSize: 58,
                        lineHeight: 0.98,
                        letterSpacing: '-0.06em',
                        fontWeight: 700,
                        maxWidth: 640,
                      }}
                    >
                      Precision-first text intelligence.
                    </div>
                    <div
                      style={{
                        marginTop: 22,
                        fontSize: 22,
                        lineHeight: 1.45,
                        color: palette.mutedStrong,
                        maxWidth: 620,
                      }}
                    >
                      A production-ready interface for sentiment, type, toxicity, irony and
                      batch review.
                    </div>
                  </div>
                  <div style={{display: 'flex', gap: 14, flexWrap: 'wrap'}}>
                    {['Sentiment', 'Type', 'Toxicity', 'Sarcasm', 'Batch mode'].map(
                      (chip, index) => (
                        <div
                          key={chip}
                          style={{
                            ...fadeUpStyle(frame, fps, 24 + index * 4, 12),
                            padding: '12px 18px',
                            borderRadius: 999,
                            background: index === 0 ? palette.blueSoft : palette.card,
                            border: `1px solid ${
                              index === 0 ? 'rgba(127,178,255,0.28)' : palette.borderSoft
                            }`,
                            color: index === 0 ? palette.text : palette.mutedStrong,
                            fontSize: 16,
                            fontWeight: 600,
                          }}
                        >
                          {chip}
                        </div>
                      ),
                    )}
                  </div>
                </Panel>

                <Panel style={{padding: 28}}>
                  <div
                    style={{
                      fontSize: 14,
                      color: palette.muted,
                      letterSpacing: '0.14em',
                      textTransform: 'uppercase',
                    }}
                  >
                    Runtime stack
                  </div>
                  <div
                    style={{
                      marginTop: 16,
                      fontSize: 40,
                      lineHeight: 0.98,
                      letterSpacing: '-0.05em',
                      fontWeight: 700,
                    }}
                  >
                    ModernBERT
                    <br />
                    Sentiment
                  </div>
                  <div
                    style={{
                      marginTop: 14,
                      fontSize: 18,
                      lineHeight: 1.5,
                      color: palette.mutedStrong,
                    }}
                  >
                    Supported by specialized models for type classification, toxicity, emotions
                    and irony detection.
                  </div>
                  <div style={{display: 'grid', gap: 12, marginTop: 30}}>
                    {modelList.slice(1).map((model, index) => (
                      <div
                        key={model}
                        style={{
                          ...fadeUpStyle(frame, fps, 32 + index * 4, 12),
                          display: 'flex',
                          justifyContent: 'space-between',
                          alignItems: 'center',
                          padding: '15px 18px',
                          borderRadius: 18,
                          background: palette.card,
                          border: `1px solid ${palette.borderSoft}`,
                        }}
                      >
                        <div style={{fontSize: 17, color: palette.mutedStrong}}>{model}</div>
                        <div
                          style={{
                            width: 8,
                            height: 8,
                            borderRadius: '50%',
                            background: index < 2 ? palette.blue : 'rgba(255,255,255,0.24)',
                          }}
                        />
                      </div>
                    ))}
                  </div>
                </Panel>
              </div>
            </BrowserFrame>
          </div>
        </AbsoluteFill>
      </SceneBackground>
    </AbsoluteFill>
  );
};

const TextScene = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const typedChars = Math.floor(
    interpolate(frame, [16, 86], [0, textComment.length], {
      extrapolateLeft: 'clamp',
      extrapolateRight: 'clamp',
    }),
  );
  const reveal = rise(frame, fps, 78, 34);

  return (
    <AbsoluteFill style={shellStyle}>
      <SceneBackground frame={frame} accentX={0.78} accentY={0.14}>
        <AbsoluteFill style={{padding: '92px 96px'}}>
          <div style={{...fadeUpStyle(frame, fps, 0), marginBottom: 34}}>
            <Kicker>Single Comment</Kicker>
            <div style={{fontSize: 74, fontWeight: 700, letterSpacing: '-0.06em'}}>
              One input. Clean output.
            </div>
          </div>
          <BrowserFrame frame={frame}>
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: '0.95fr 1.05fr',
                gap: 24,
                padding: 28,
                height: '100%',
              }}
            >
              <Panel style={{padding: 28}}>
                <div style={{fontSize: 30, fontWeight: 650, letterSpacing: '-0.04em'}}>
                  Classify a comment
                </div>
                <div
                  style={{
                    marginTop: 10,
                    fontSize: 18,
                    color: palette.mutedStrong,
                    lineHeight: 1.45,
                  }}
                >
                  Enter any comment and reveal sentiment, type and toxicity in one pass.
                </div>
                <div
                  style={{
                    marginTop: 24,
                    minHeight: 278,
                    padding: 24,
                    borderRadius: 24,
                    background: 'rgba(0,0,0,0.28)',
                    border: `1px solid ${palette.borderSoft}`,
                    fontSize: 28,
                    lineHeight: 1.42,
                    color: palette.text,
                  }}
                >
                  {textComment.slice(0, typedChars)}
                  <span style={{opacity: frame % 18 < 9 ? 1 : 0, color: palette.blue}}>|</span>
                </div>
                <div style={{display: 'flex', gap: 14, marginTop: 18}}>
                  <div
                    style={{
                      flex: 1,
                      padding: '18px 22px',
                      borderRadius: 18,
                      background: 'linear-gradient(180deg, #8ab9ff 0%, #5b96f0 100%)',
                      color: '#031221',
                      fontSize: 20,
                      fontWeight: 700,
                      textAlign: 'center',
                    }}
                  >
                    Classify
                  </div>
                  <div
                    style={{
                      padding: '18px 22px',
                      borderRadius: 18,
                      background: palette.card,
                      border: `1px solid ${palette.borderSoft}`,
                      fontSize: 20,
                      color: palette.mutedStrong,
                    }}
                  >
                    Clear
                  </div>
                </div>
              </Panel>

              <Panel
                style={{
                  padding: 28,
                  opacity: reveal,
                  transform: `translateY(${interpolate(reveal, [0, 1], [24, 0])}px)`,
                }}
              >
                <div
                  style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center'}}
                >
                  <div>
                    <div
                      style={{
                        fontSize: 15,
                        color: palette.muted,
                        letterSpacing: '0.14em',
                        textTransform: 'uppercase',
                      }}
                    >
                      Result
                    </div>
                    <div
                      style={{
                        marginTop: 10,
                        fontSize: 44,
                        fontWeight: 700,
                        letterSpacing: '-0.05em',
                      }}
                    >
                      Negative
                    </div>
                  </div>
                  <div
                    style={{
                      padding: '10px 14px',
                      borderRadius: 999,
                      background: 'rgba(255,255,255,0.06)',
                      border: `1px solid ${palette.borderSoft}`,
                      fontSize: 14,
                      color: palette.mutedStrong,
                    }}
                  >
                    84ms
                  </div>
                </div>

                <div style={{display: 'flex', gap: 12, marginTop: 20}}>
                  {[
                    ['Complaint', palette.text],
                    ['Toxic', palette.red],
                    ['High confidence', palette.blue],
                  ].map(([label, color]) => (
                    <div
                      key={label}
                      style={{
                        padding: '10px 16px',
                        borderRadius: 999,
                        background: palette.card,
                        border: `1px solid ${palette.borderSoft}`,
                        color,
                        fontSize: 15,
                        fontWeight: 600,
                      }}
                    >
                      {label}
                    </div>
                  ))}
                </div>

                <div style={{display: 'flex', gap: 14, marginTop: 24}}>
                  <MetricChip label="Sentiment engine" value="ModernBERT" color={palette.blue} />
                  <MetricChip label="Toxicity" value="Detected" color={palette.red} />
                  <MetricChip label="Comment type" value="Complaint" />
                </div>

                <div style={{marginTop: 26, display: 'grid', gap: 16}}>
                  <ConfidenceRow label="Positive" pct={3} color={palette.green} />
                  <ConfidenceRow label="Neutral" pct={8} color={palette.amber} />
                  <ConfidenceRow label="Negative" pct={97} color={palette.red} />
                </div>

                <div
                  style={{
                    marginTop: 30,
                    paddingTop: 20,
                    borderTop: `1px solid ${palette.borderSoft}`,
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                  }}
                >
                  <div style={{fontSize: 16, color: palette.mutedStrong}}>
                    Supporting models aligned in the background.
                  </div>
                  <div style={{fontSize: 14, color: palette.muted}}>
                    Toxic-BERT · BART MNLI · Irony
                  </div>
                </div>
              </Panel>
            </div>
          </BrowserFrame>
        </AbsoluteFill>
      </SceneBackground>
    </AbsoluteFill>
  );
};

const BatchScene = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const progress = interpolate(frame, [18, 96], [10, 100], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
    easing: Easing.out(Easing.cubic),
  });
  const tableReveal = rise(frame, fps, 92, 34);

  return (
    <AbsoluteFill style={shellStyle}>
      <SceneBackground frame={frame} accentX={0.22} accentY={0.78}>
        <AbsoluteFill style={{padding: '92px 96px'}}>
          <div style={{...fadeUpStyle(frame, fps, 0), marginBottom: 34}}>
            <Kicker>Batch Workflow</Kicker>
            <div style={{fontSize: 74, fontWeight: 700, letterSpacing: '-0.06em'}}>
              Upload once. Review fast.
            </div>
          </div>
          <BrowserFrame frame={frame}>
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: '0.84fr 1.16fr',
                gap: 24,
                padding: 28,
                height: '100%',
              }}
            >
              <Panel style={{padding: 28}}>
                <div style={{fontSize: 30, fontWeight: 650, letterSpacing: '-0.04em'}}>
                  Batch classification
                </div>
                <div
                  style={{
                    marginTop: 10,
                    fontSize: 18,
                    color: palette.mutedStrong,
                    lineHeight: 1.45,
                  }}
                >
                  Bring in a CSV and review sentiment, type and edge cases at scale.
                </div>
                <div
                  style={{
                    marginTop: 24,
                    padding: '52px 26px',
                    borderRadius: 26,
                    background: 'rgba(255,255,255,0.03)',
                    border: `1px dashed ${palette.border}`,
                    textAlign: 'center',
                  }}
                >
                  <div
                    style={{
                      fontSize: 18,
                      color: palette.muted,
                      letterSpacing: '0.16em',
                      textTransform: 'uppercase',
                    }}
                  >
                    Upload
                  </div>
                  <div
                    style={{
                      marginTop: 18,
                      fontSize: 30,
                      fontWeight: 600,
                      letterSpacing: '-0.04em',
                    }}
                  >
                    customer-comments-march.csv
                  </div>
                  <div style={{marginTop: 10, fontSize: 16, color: palette.mutedStrong}}>
                    1,284 rows · 3.4 MB
                  </div>
                </div>

                <div style={{marginTop: 22}}>
                  <div
                    style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      marginBottom: 10,
                      fontSize: 15,
                      color: palette.mutedStrong,
                    }}
                  >
                    <span>Processing</span>
                    <span>{Math.round(progress)}%</span>
                  </div>
                  <div
                    style={{
                      height: 8,
                      borderRadius: 999,
                      background: 'rgba(255,255,255,0.06)',
                      overflow: 'hidden',
                    }}
                  >
                    <div
                      style={{
                        width: `${progress}%`,
                        height: '100%',
                        borderRadius: 999,
                        background: 'linear-gradient(90deg, #8ab9ff 0%, #5a95f1 100%)',
                      }}
                    />
                  </div>
                </div>

                <div
                  style={{display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginTop: 24}}
                >
                  <MetricChip label="Top type" value="Complaint" />
                  <MetricChip label="Low confidence" value="17" color={palette.amber} />
                  <MetricChip label="Sarcasm flags" value="24" />
                  <MetricChip label="Toxic rows" value="38" color={palette.red} />
                </div>
              </Panel>

              <Panel
                style={{
                  padding: 24,
                  opacity: tableReveal,
                  transform: `translateY(${interpolate(tableReveal, [0, 1], [20, 0])}px)`,
                }}
              >
                <div
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    marginBottom: 18,
                  }}
                >
                  <div>
                    <div style={{fontSize: 28, fontWeight: 650, letterSpacing: '-0.04em'}}>
                      Batch results
                    </div>
                    <div style={{marginTop: 6, fontSize: 16, color: palette.mutedStrong}}>
                      Review rows, scan confidence, export cleanly.
                    </div>
                  </div>
                  <div
                    style={{
                      padding: '14px 18px',
                      borderRadius: 18,
                      background: palette.cardStrong,
                      border: `1px solid ${palette.borderSoft}`,
                      fontSize: 16,
                      color: palette.text,
                      fontWeight: 600,
                    }}
                  >
                    Export CSV
                  </div>
                </div>

                <div
                  style={{
                    borderRadius: 24,
                    overflow: 'hidden',
                    border: `1px solid ${palette.borderSoft}`,
                    background: 'rgba(255,255,255,0.03)',
                  }}
                >
                  <div
                    style={{
                      display: 'grid',
                      gridTemplateColumns: '1.25fr 0.42fr 0.42fr 0.32fr',
                      padding: '16px 18px',
                      fontSize: 12,
                      textTransform: 'uppercase',
                      letterSpacing: '0.14em',
                      color: palette.muted,
                      background: 'rgba(255,255,255,0.04)',
                    }}
                  >
                    <div>Comment</div>
                    <div>Sentiment</div>
                    <div>Type</div>
                    <div>Confidence</div>
                  </div>
                  {batchRows.map(([comment, sentiment, type, confidence], index) => (
                    <div
                      key={comment}
                      style={{
                        display: 'grid',
                        gridTemplateColumns: '1.25fr 0.42fr 0.42fr 0.32fr',
                        gap: 14,
                        padding: '17px 18px',
                        fontSize: 15,
                        lineHeight: 1.45,
                        borderTop: index === 0 ? 'none' : `1px solid ${palette.borderSoft}`,
                      }}
                    >
                      <div style={{paddingRight: 18, color: palette.text}}>{comment}</div>
                      <div
                        style={{
                          color:
                            sentiment === 'Positive'
                              ? palette.green
                              : sentiment === 'Negative'
                                ? palette.red
                                : palette.amber,
                          fontWeight: 600,
                        }}
                      >
                        {sentiment}
                      </div>
                      <div style={{color: palette.mutedStrong}}>{type}</div>
                      <div style={{color: palette.muted, fontVariantNumeric: 'tabular-nums'}}>
                        {confidence}
                      </div>
                    </div>
                  ))}
                </div>
              </Panel>
            </div>
          </BrowserFrame>
        </AbsoluteFill>
      </SceneBackground>
    </AbsoluteFill>
  );
};

const ClosingScene = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();

  return (
    <AbsoluteFill style={shellStyle}>
      <SceneBackground frame={frame} accentX={0.5} accentY={0.1}>
        <AbsoluteFill
          style={{
            padding: '108px 110px',
            justifyContent: 'space-between',
          }}
        >
          <div style={{...fadeUpStyle(frame, fps, 0), maxWidth: 940}}>
            <Kicker>Closing Frame</Kicker>
            <div
              style={{
                fontSize: 88,
                lineHeight: 0.92,
                fontWeight: 700,
                letterSpacing: '-0.07em',
              }}
            >
              Powerful analysis.
              <br />
              Effortless presentation.
            </div>
            <div
              style={{
                marginTop: 26,
                fontSize: 28,
                lineHeight: 1.4,
                color: palette.mutedStrong,
                maxWidth: 780,
              }}
            >
              Smart Comment Classification pairs a calm interface with a robust NLP stack built
              for real production workflows.
            </div>
          </div>

          <div
            style={{
              display: 'grid',
              gridTemplateColumns: '1.15fr 0.85fr',
              gap: 24,
              alignItems: 'end',
            }}
          >
            <Panel style={{padding: 26}}>
              <div
                style={{
                  fontSize: 14,
                  color: palette.muted,
                  textTransform: 'uppercase',
                  letterSpacing: '0.14em',
                }}
              >
                Model stack
              </div>
              <div style={{display: 'flex', flexWrap: 'wrap', gap: 12, marginTop: 18}}>
                {modelList.map((model, index) => (
                  <div
                    key={model}
                    style={{
                      ...fadeUpStyle(frame, fps, 18 + index * 3, 12),
                      padding: '10px 14px',
                      borderRadius: 999,
                      background: index === 0 ? palette.blueSoft : palette.card,
                      border: `1px solid ${
                        index === 0 ? 'rgba(127,178,255,0.26)' : palette.borderSoft
                      }`,
                      fontSize: 16,
                      color: index === 0 ? palette.text : palette.mutedStrong,
                    }}
                  >
                    {model}
                  </div>
                ))}
              </div>
            </Panel>

            <div style={{...fadeUpStyle(frame, fps, 14), justifySelf: 'end'}}>
              <div
                style={{
                  padding: '18px 24px',
                  borderRadius: 999,
                  background: 'rgba(255,255,255,0.08)',
                  border: `1px solid ${palette.border}`,
                  fontSize: 24,
                  fontWeight: 650,
                  letterSpacing: '-0.04em',
                }}
              >
                Smart Comment Classification
              </div>
            </div>
          </div>
        </AbsoluteFill>
      </SceneBackground>
    </AbsoluteFill>
  );
};

export const ProductDemo = ({title, subtitle}) => {
  return (
    <AbsoluteFill style={shellStyle}>
      <TransitionSeries>
        <TransitionSeries.Sequence durationInFrames={250}>
          <IntroScene title={title} subtitle={subtitle} />
        </TransitionSeries.Sequence>
        <TransitionSeries.Transition
          presentation={fade()}
          timing={linearTiming({durationInFrames: 20})}
        />
        <TransitionSeries.Sequence durationInFrames={220}>
          <TextScene />
        </TransitionSeries.Sequence>
        <TransitionSeries.Transition
          presentation={fade()}
          timing={linearTiming({durationInFrames: 20})}
        />
        <TransitionSeries.Sequence durationInFrames={220}>
          <BatchScene />
        </TransitionSeries.Sequence>
        <TransitionSeries.Transition
          presentation={fade()}
          timing={linearTiming({durationInFrames: 20})}
        />
        <TransitionSeries.Sequence durationInFrames={270}>
          <ClosingScene />
        </TransitionSeries.Sequence>
      </TransitionSeries>
    </AbsoluteFill>
  );
};
